//! Final Cut Pro 7 XML interchange (`xmeml` versions 4 and 5), the dialect Premiere Pro imports
//! and exports.
//!
//! Supported: `project`/`bin`/`clip`/`sequence` structure, video and audio tracks (enabled,
//! locked), `clipitem` (start/end/in/out, `pproTicksIn/Out` for tick-exact source times, enabled,
//! labels, markers, links), master-clip `<clip>` media markers, `file` (pathurl, rate, duration,
//! timecode, media characteristics, references by id), nested sequences (inline or by id),
//! `generatoritem` (Slug, Color, …),
//! `transitionitem` (alignment, Cross Dissolve & friends), and the filters Basic Motion (scale,
//! rotation, center, anchor, anti-flicker), Opacity, Time Remap (constant speed, reverse, freeze),
//! Audio Levels and Audio Pan, all with keyframes. Other effects are written with our effect ids
//! and read back by id or display name; anything unknown is reported.
//!
//! Stereo: a clip of a two-channel file is written as Premiere writes it, two mono clip items
//! (`sourcetrack/trackindex` 1 and 2) on a pair of XML tracks (`outputchannelindex` 1 and 2),
//! linked with one `groupindex`, each with an Audio Pan hard to its side; FCP7 XML has no balance
//! for a stereo pair, so the clip's Balance becomes the halves' levels. Import reads such a pair
//! (same file, times and effects, panned to their sides) back as one stereo clip.
//!
//! Coordinates: Basic Motion `center` is written relative to the frame centre as a fraction of the
//! frame width/height (0,0 = centred); `centerOffset` (anchor) likewise relative to the source.
//! Keyframe `when` values and clip markers are source (media) frames, like `in`/`out`.

use std::collections::{HashMap, HashSet};

use filmcraft_geom::Vec2;
use filmcraft_media::{DemoScene, Generator, MediaKind};
use filmcraft_project::{
    BinId, ClipId, EffectInstance, Interpolation, ItemId, ItemKind, Keyframe, Label, Marker, MarkerId, MarkerKind, Param, ParamValue, Project, Sequence,
    TrackItem, TrackKind, Transition, TransitionAlign, TransitionId, effect::find_effect_by_name, find_effect,
};
use filmcraft_time::{FrameRate, Tick, format_timecode_frames};
use roxmltree::Node;

use crate::common::{
    Builder, MediaSpec, balance_for, balance_gains, base_item, db_to_gain, empty_sequence, file_name, frames_round, gain_to_db, item_media, item_path,
    kind_from_ext, label_from, on_frame, param, param_modified, path_to_file_url, rate_from_timebase, resolve_path, set_param, settings_for, standard_effects,
    timebase, transition_effect, transition_name, unscaled,
};
use crate::xml::{XmlWriter, bool_str, child, child_bool, child_f64, child_i64, child_text, children, elements, num, path_text};
use crate::{Error, ExportOptions, Format, ImportOptions, Imported, Report, Result};

// ---------------------------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------------------------

const BLEND_XML: &[(&str, u32)] = &[
    ("normal", 0),
    ("darken", 2),
    ("multiply", 3),
    ("lighten", 7),
    ("screen", 8),
    ("add", 10),
    ("overlay", 12),
    ("softlight", 13),
    ("hardlight", 14),
    ("difference", 19),
    ("subtract", 21),
];

fn generator_from(name: &str, id: &str, report: &mut Report) -> Generator {
    if let Some(json) = id.strip_prefix("filmcraft:")
        && let Ok(g) = serde_json::from_str::<Generator>(json)
    {
        return g;
    }
    match name.to_ascii_lowercase().as_str() {
        "slug" | "black video" => Generator::BlackVideo,
        "color" | "color matte" | "colour matte" => Generator::ColorMatte { color: [0.0, 0.0, 0.0, 1.0] },
        "transparent video" => Generator::TransparentVideo,
        "counting leader" | "universal counting leader" => Generator::CountingLeader,
        n if n.starts_with("bars and tone") => Generator::BarsAndTone,
        n => {
            if let Some(d) = DemoScene::ALL.iter().find(|d| d.file_name().eq_ignore_ascii_case(n)) {
                return Generator::Demo(*d);
            }
            report.warn(format!("generator \"{name}\" is not supported; imported as Black Video"));
            Generator::BlackVideo
        }
    }
}

fn read_rate(n: Node) -> Option<FrameRate> {
    let r = child(n, "rate")?;
    let tb = child_i64(r, "timebase")?;
    Some(rate_from_timebase(tb, child_bool(r, "ntsc").unwrap_or(false)))
}

fn write_rate(w: &mut XmlWriter, rate: FrameRate) {
    let (tb, ntsc) = timebase(rate);
    w.open("rate", &[]);
    w.text("timebase", tb);
    w.text("ntsc", bool_str(ntsc));
    w.close();
}

// ---------------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------------

struct PendingLink {
    seq: ItemId,
    kind: TrackKind,
    track: usize,
    clip: ClipId,
    refs: Vec<String>,
}

struct Imp<'a, 'i, 'r> {
    b: Builder,
    opts: &'a ImportOptions,
    report: &'r mut Report,
    files: HashMap<String, Node<'a, 'i>>,
    seq_defs: HashMap<String, Node<'a, 'i>>,
    file_items: HashMap<String, ItemId>,
    seq_items: HashMap<String, ItemId>,
    in_progress: HashSet<String>,
    links: Vec<PendingLink>,
    xml_clip_ids: HashMap<(ItemId, String), ClipId>,
    /// `(sequence key, file key)` pairs where some audio clip item of that file uses `sourcetrack/trackindex` > 1,
    /// i.e. the file's channels are split onto mono tracks in that sequence (#463).
    split_files: HashSet<(String, String)>,
}

/// The `(sequence key, file key)` an audio clip item belongs to, if it references a file.
fn clip_seq_file(c: Node) -> Option<(String, String)> {
    let seq = c.ancestors().skip(1).find(|a| a.has_tag_name("sequence"))?;
    Some((node_key(seq), node_key(child(c, "file")?)))
}

/// The 1-based `sourcetrack/trackindex` of an audio clip item.
fn audio_track_index(c: Node) -> Option<i64> {
    let st = child(c, "sourcetrack")?;
    if child_text(st, "mediatype").is_some_and(|m| m != "audio") {
        return None;
    }
    child_i64(st, "trackindex")
}

fn node_key(n: Node) -> String {
    match n.attribute("id") {
        Some(id) => id.to_string(),
        None => format!("#node{}", n.id().get()),
    }
}

pub(crate) fn import(text_in: &str, opts: &ImportOptions, report: &mut Report) -> Result<Imported> {
    let doc = crate::xml::parse(text_in).map_err(|e| Error::parse(Format::Fcp7Xml, e))?;
    let root = doc.root_element();
    if root.tag_name().name() != "xmeml" {
        return Err(Error::parse(Format::Fcp7Xml, "root element is not <xmeml>"));
    }
    let mut imp = Imp {
        b: Builder::new(root.descendants().find(|n| n.has_tag_name("project")).and_then(|p| child_text(p, "name")).unwrap_or("Imported XML")),
        opts,
        report,
        files: HashMap::new(),
        seq_defs: HashMap::new(),
        file_items: HashMap::new(),
        seq_items: HashMap::new(),
        in_progress: HashSet::new(),
        links: Vec::new(),
        xml_clip_ids: HashMap::new(),
        split_files: HashSet::new(),
    };
    for n in root.descendants().filter(|n| n.is_element()) {
        match n.tag_name().name() {
            "clipitem" if audio_track_index(n).is_some_and(|i| i > 1) => {
                if let Some(key) = clip_seq_file(n) {
                    imp.split_files.insert(key);
                }
            }
            "file" if elements(n).next().is_some() => {
                imp.files.entry(node_key(n)).or_insert(n);
            }
            "sequence" if child(n, "media").is_some() => {
                imp.seq_defs.entry(node_key(n)).or_insert(n);
            }
            _ => {}
        }
    }
    imp.walk_container(root, None, true);
    imp.resolve_links();
    let seqs: Vec<ItemId> = imp.seq_items.values().copied().collect();
    for s in seqs {
        imp.merge_stereo_halves(s);
    }
    if imp.b.p.items.is_empty() {
        return Err(Error::Empty);
    }
    Ok(imp.b.finish())
}

impl<'a, 'i> Imp<'a, 'i, '_> {
    /// Walk `xmeml`, `project`, `bin` and `children` containers.
    fn walk_container(&mut self, n: Node<'a, 'i>, bin: Option<BinId>, top: bool) {
        for c in elements(n) {
            match c.tag_name().name() {
                "project" => {
                    let target = c.children().find(|x| x.has_tag_name("children")).unwrap_or(c);
                    self.walk_container(target, bin, top);
                }
                "children" => self.walk_container(c, bin, top),
                "bin" => {
                    let name = child_text(c, "name").unwrap_or("Bin");
                    let b = self.b.bin(name, bin);
                    let target = child(c, "children").unwrap_or(c);
                    self.walk_container(target, Some(b), top);
                }
                "sequence" => {
                    if let Some(id) = self.sequence(c, bin)
                        && top
                        && !self.b.top.contains(&id)
                    {
                        self.b.top.push(id);
                    }
                }
                "clip" => self.master_clip(c, bin),
                _ => {}
            }
        }
    }

    /// A master clip in a bin: its file(s) become media items.
    fn master_clip(&mut self, n: Node<'a, 'i>, bin: Option<BinId>) {
        let file = n.descendants().find(|d| d.has_tag_name("file"));
        match file {
            Some(f) => {
                let id = self.file(f, child_text(n, "name"), bin);
                // A sequence's clip items file their media at the top level, so a sequence earlier in the
                // document that used this file first left it there; the master clip's bin is its home.
                if let (Some(item), Some(b)) = (id, bin)
                    && self.b.p.root.parent_of(item) == Some(self.b.p.root.id)
                {
                    self.b.p.move_to_bin(&[item], Some(b));
                }
                if let Some(item) = id.and_then(|i| self.b.p.item_mut(i))
                    && let Some(l) = path_text(n, &["labels", "label2"]).and_then(label_from)
                {
                    item.label = l;
                }
                if let (Some(item), Some(rate)) = (id, read_rate(n)) {
                    let markers = read_markers(n, rate, &mut self.b);
                    if let Some(m) = self.b.p.item_mut(item).and_then(|i| i.as_media_mut())
                        && m.markers.is_empty()
                    {
                        m.markers = markers;
                    }
                }
            }
            None => {
                if let Some(s) = n.descendants().find(|d| d.has_tag_name("sequence")) {
                    self.sequence(s, bin);
                }
            }
        }
    }

    /// Media item for a `<file>` element (definition or reference by id).
    fn file(&mut self, n: Node<'a, 'i>, fallback_name: Option<&str>, bin: Option<BinId>) -> Option<ItemId> {
        let key = node_key(n);
        if let Some(id) = self.file_items.get(&key) {
            return Some(*id);
        }
        let def = if elements(n).next().is_some() { n } else { *self.files.get(&key)? };
        let name = child_text(def, "name").filter(|s| !s.is_empty()).or(fallback_name).unwrap_or("Untitled").to_string();
        let rate = read_rate(def);
        let video = child(def, "media").and_then(|m| child(m, "video"));
        let audio = child(def, "media").and_then(|m| child(m, "audio"));
        let dims = video.map(|v| {
            let sc = child(v, "samplecharacteristics").unwrap_or(v);
            (child_i64(sc, "width").unwrap_or(1920) as u32, child_i64(sc, "height").unwrap_or(1080) as u32)
        });
        // Premiere describes a file's sound as one `<audio>` per source channel (`channelcount` 1
        // each, the stereo pair of our QuickTime files as two), so the channels are their sum.
        let channels: i64 = child(def, "media").map_or(0, |m| children(m, "audio").map(|a| child_i64(a, "channelcount").unwrap_or(2).clamp(1, 64)).sum());
        let aud = audio.map(|a| {
            let sc = child(a, "samplecharacteristics").unwrap_or(a);
            (child_i64(sc, "samplerate").unwrap_or(48_000) as u32, channels.clamp(1, 64) as u32)
        });
        let r = rate.unwrap_or_default();
        let path = child_text(def, "pathurl").filter(|s| !s.is_empty());
        let mut spec = MediaSpec {
            duration: child_i64(def, "duration").filter(|d| *d > 0).map(|d| r.tick_of(d)),
            video: dims.map(|(w, h)| (w, h, r)),
            audio: aud,
            start_tc: child(def, "timecode").and_then(|t| child_i64(t, "frame")),
            kind: None,
        };
        if spec.video.is_none() && spec.audio.is_none() {
            // No media characteristics: guess from the extension.
            match kind_from_ext(path.unwrap_or(&name)) {
                MediaKind::AudioOnly => spec.audio = Some((48_000, 2)),
                k => {
                    spec.video = Some((1920, 1080, r));
                    spec.kind = Some(k);
                }
            }
        }
        let id = match path {
            Some(url) => {
                let p = resolve_path(url, self.opts.base_dir.as_deref());
                self.b.file_media(&format!("file:{p}"), &name, &p, &spec, bin)
            }
            None => {
                self.report.warn(format!("file \"{name}\" has no pathurl; imported offline"));
                let p = resolve_path(&name, self.opts.base_dir.as_deref());
                let id = self.b.file_media(&format!("fileid:{key}"), &name, &p, &spec, bin);
                if let Some(m) = self.b.p.item_mut(id).and_then(|i| i.as_media_mut()) {
                    m.offline = true;
                }
                id
            }
        };
        self.file_items.insert(key, id);
        Some(id)
    }

    /// Sequence item for a `<sequence>` element (definition or reference).
    fn sequence(&mut self, n: Node<'a, 'i>, bin: Option<BinId>) -> Option<ItemId> {
        let key = node_key(n);
        if let Some(id) = self.seq_items.get(&key) {
            return Some(*id);
        }
        let def = if child(n, "media").is_some() { n } else { *self.seq_defs.get(&key)? };
        if !self.in_progress.insert(key.clone()) {
            self.report.warn("recursive nested sequence ignored");
            return None;
        }
        let name = child_text(def, "name").unwrap_or("Sequence").to_string();
        let rate = read_rate(def).unwrap_or_default();
        let tc = child(def, "timecode");
        let df = tc.and_then(|t| child_text(t, "displayformat")).is_some_and(|d| d.eq_ignore_ascii_case("DF"));
        let vfmt = child(def, "media").and_then(|m| child(m, "video")).and_then(|v| child(v, "format")).and_then(|f| child(f, "samplecharacteristics"));
        let (w, h) = vfmt.map(|s| (child_i64(s, "width").unwrap_or(1920) as u32, child_i64(s, "height").unwrap_or(1080) as u32)).unwrap_or((1920, 1080));
        let mut settings = settings_for(rate, w, h, df);
        if let Some(sr) = child(def, "media")
            .and_then(|m| child(m, "audio"))
            .and_then(|a| child(a, "format"))
            .and_then(|f| child(f, "samplecharacteristics"))
            .and_then(|s| child_i64(s, "samplerate"))
        {
            settings.sample_rate = sr as u32;
        }
        let id = self.b.reserve_sequence(&name, settings.clone(), bin);
        if let Some(l) = path_text(def, &["labels", "label2"]).and_then(label_from)
            && let Some(it) = self.b.p.item_mut(id)
        {
            it.label = l;
        }
        self.seq_items.insert(key.clone(), id);
        let mut seq = empty_sequence(settings);
        seq.start_timecode = tc.and_then(|t| child_i64(t, "frame")).unwrap_or(0);
        seq.markers = read_markers(def, rate, &mut self.b);
        let media = child(def, "media");
        for (kind, tag) in [(TrackKind::Video, "video"), (TrackKind::Audio, "audio")] {
            let Some(m) = media.and_then(|m| child(m, tag)) else { continue };
            for (ti, tn) in children(m, "track").enumerate() {
                self.b.ensure_tracks(&mut seq, kind, ti + 1);
                self.track(&mut seq, id, kind, ti, tn, rate, (w, h));
            }
        }
        if seq.video_tracks.is_empty() {
            self.b.ensure_tracks(&mut seq, TrackKind::Video, 1);
        }
        if seq.audio_tracks.is_empty() {
            self.b.ensure_tracks(&mut seq, TrackKind::Audio, 1);
        }
        self.b.put_sequence(id, seq);
        self.in_progress.remove(&key);
        Some(id)
    }

    #[allow(clippy::too_many_arguments)]
    fn track(&mut self, seq: &mut Sequence, seq_id: ItemId, kind: TrackKind, ti: usize, tn: Node<'a, 'i>, rate: FrameRate, frame: (u32, u32)) {
        {
            let t = &mut seq.tracks_mut(kind)[ti];
            t.enabled = child_bool(tn, "enabled").unwrap_or(true);
            t.locked = child_bool(tn, "locked").unwrap_or(false);
        }
        let mut trans_nodes = Vec::new();
        for c in elements(tn) {
            match c.tag_name().name() {
                "clipitem" | "generatoritem" => {
                    if let Some((ti_item, refs)) = self.clip_item(c, kind, rate, frame) {
                        let xml_id = node_key(c);
                        self.xml_clip_ids.insert((seq_id, xml_id), ti_item.id);
                        if !refs.is_empty() {
                            self.links.push(PendingLink { seq: seq_id, kind, track: ti, clip: ti_item.id, refs });
                        }
                        let t = &mut seq.tracks_mut(kind)[ti];
                        if t.items.iter().any(|o| o.start < ti_item.end() && ti_item.start < o.end()) {
                            self.report.warn("overlapping clip items on one track; the later one was moved to a new track");
                            self.b.place(seq, kind, ti + 1, ti_item);
                        } else {
                            t.items.push(ti_item);
                        }
                    }
                }
                "transitionitem" => trans_nodes.push(c),
                _ => {}
            }
        }
        seq.tracks_mut(kind)[ti].sort();
        for c in trans_nodes {
            self.transition(seq, kind, ti, c, rate);
        }
    }

    fn clip_item(&mut self, c: Node<'a, 'i>, kind: TrackKind, seq_rate: FrameRate, frame: (u32, u32)) -> Option<(TrackItem, Vec<String>)> {
        let name = child_text(c, "name").unwrap_or("").to_string();
        let rate = read_rate(c).unwrap_or(seq_rate);
        let item = if c.has_tag_name("generatoritem") {
            let eff = child(c, "effect");
            let gname = eff.and_then(|e| child_text(e, "name")).unwrap_or(&name).to_string();
            let gid = eff.and_then(|e| child_text(e, "effectid")).unwrap_or("").to_string();
            let mut g = generator_from(&gname, &gid, self.report);
            if let (Generator::ColorMatte { color }, Some(e)) = (&mut g, eff)
                && let Some(v) = e.descendants().find(|d| d.has_tag_name("value") && child(*d, "red").is_some())
            {
                *color = read_color(v);
            }
            let spec = MediaSpec { video: Some((frame.0, frame.1, seq_rate)), ..Default::default() };
            let key = format!("gen:{}", serde_json::to_string(&g).unwrap_or_default());
            let label = if name.is_empty() { g.label() } else { name.clone() };
            self.b.generator_media(&key, &label, g, &spec, None)
        } else if let Some(f) = child(c, "file") {
            match self.file(f, Some(&name), None) {
                Some(item) => item,
                None => {
                    self.report.warn(format!("clip \"{name}\" refers to a file the document does not define; skipped"));
                    return None;
                }
            }
        } else if let Some(s) = child(c, "sequence") {
            self.sequence(s, None)?
        } else {
            self.report.warn(format!("clip \"{name}\" has no media reference; imported offline"));
            let spec = MediaSpec {
                video: (kind == TrackKind::Video).then_some((frame.0, frame.1, seq_rate)),
                audio: (kind == TrackKind::Audio).then_some((48_000, 2)),
                ..Default::default()
            };
            let key = format!("offline:{name}");
            let p = resolve_path(&name, self.opts.base_dir.as_deref());
            let id = self.b.file_media(&key, &name, &p, &spec, None);
            if let Some(m) = self.b.p.item_mut(id).and_then(|i| i.as_media_mut()) {
                m.offline = true;
            }
            id
        };

        let start = child_i64(c, "start").unwrap_or(-1);
        let end = child_i64(c, "end").unwrap_or(-1);
        let in_f = child_i64(c, "in").unwrap_or(0);
        let out_f = child_i64(c, "out").unwrap_or(in_f);
        let src_in = child_i64(c, "pproTicksIn").map(Tick).unwrap_or_else(|| rate.tick_of(in_f));
        let src_out = child_i64(c, "pproTicksOut").map(Tick).unwrap_or_else(|| rate.tick_of(out_f));

        // Speed from the Time Remap filter (or the in/out vs start/end ratio).
        let mut speed = 1.0;
        let mut reverse = false;
        let mut hold = false;
        for eff in c.children().filter(|x| x.has_tag_name("filter")).filter_map(|f| child(f, "effect")) {
            if child_text(eff, "effectid") == Some("timeremap") {
                for p in children(eff, "parameter") {
                    match child_text(p, "parameterid") {
                        Some("speed") => {
                            let v = child_f64(p, "value").unwrap_or(100.0);
                            if v == 0.0 {
                                hold = true;
                            } else {
                                speed = v.abs() / 100.0;
                                reverse |= v < 0.0;
                            }
                            if child(p, "keyframe").is_some() {
                                self.report.warn("variable speed (time remap keyframes) is not supported; a constant speed was used");
                            }
                        }
                        Some("reverse") => reverse |= child_bool(p, "value").unwrap_or(false),
                        Some("variablespeed") if child_bool(p, "value") == Some(true) || child_f64(p, "value").is_some_and(|v| v != 0.0) => {
                            self.report.warn("variable speed (time remap keyframes) is not supported; a constant speed was used");
                        }
                        _ => {}
                    }
                }
            }
        }
        let src_len = (src_out - src_in).abs();
        let (start_t, dur) = match (start >= 0, end >= 0) {
            (true, true) => (seq_rate.tick_of(start), seq_rate.tick_of(end) - seq_rate.tick_of(start)),
            (true, false) => {
                let d = if hold { rate.tick_of(out_f - in_f) } else { unscaled(src_len, speed) };
                (seq_rate.tick_of(start), d)
            }
            (false, true) => {
                let d = if hold { rate.tick_of(out_f - in_f) } else { unscaled(src_len, speed) };
                (seq_rate.tick_of(end) - d, d)
            }
            (false, false) => {
                self.report.warn(format!("clip \"{name}\" has neither start nor end; skipped"));
                return None;
            }
        };
        if dur <= Tick::ZERO {
            return None;
        }
        let has_remap =
            c.children().filter(|x| x.has_tag_name("filter")).filter_map(|f| child(f, "effect")).any(|e| child_text(e, "effectid") == Some("timeremap"));
        if !has_remap && !hold && src_len != dur && src_len > Tick::ZERO {
            speed = (src_len.0 as f64 / dur.0 as f64 * 10_000.0).round() / 10_000.0;
        }
        let mut ti = self.b.clip(item, kind, if name.is_empty() { "Clip" } else { &name }, start_t, dur, src_in.min(src_out));
        ti.speed = speed;
        ti.reverse = reverse;
        if hold {
            ti.frame_hold = Some(src_in);
        }
        ti.enabled = child_bool(c, "enabled").unwrap_or(true);
        if let Some(l) = path_text(c, &["labels", "label2"]).and_then(label_from) {
            ti.label = l;
        }
        // `sourcetrack/trackindex` (1-based) picks the source channel of a mono track split off a multichannel file (#463).
        // Index 1 is also what a stereo clip on a stereo track carries, so it means channel 0 only when another clip
        // item of the same file in this sequence uses a higher index (the file is split); otherwise it keeps the
        // default (empty) channel mapping.
        if kind == TrackKind::Audio
            && let Some(idx) = audio_track_index(c)
            && (idx > 1 || (idx == 1 && clip_seq_file(c).is_some_and(|k| self.split_files.contains(&k))))
            && let Ok(ch) = u16::try_from(idx - 1)
        {
            ti.source_channels = vec![ch];
            // The two halves of a split stereo file play on their own side, as the stereo clip did (#371):
            // a mono clip is centred (both sides at unity), so the halves would otherwise sum on both
            // sides, +3 dB for uncorrelated channels, +6 dB for dual mono. An `audiopan` filter
            // (read below) still overrides this.
            let stereo = matches!(&self.b.p.item(item).map(|i| &i.kind), Some(ItemKind::Media(m)) if m.info.audio().is_some_and(|a| a.channels == 2));
            if stereo && ch <= 1 {
                set_param(&mut ti, "panner", "balance", Param::new(ParamValue::Float(if ch == 0 { -100.0 } else { 100.0 })));
            }
        }
        if let Some(mode) = child_text(c, "compositemode") {
            match BLEND_XML.iter().find(|(n, _)| n.eq_ignore_ascii_case(mode)) {
                Some((_, v)) if *v != 0 => set_param(&mut ti, "opacity", "blend", Param::new(ParamValue::Choice(*v))),
                Some(_) => {}
                None => self.report.warn(format!("composite mode \"{mode}\" is not supported")),
            }
        }
        ti.markers = read_markers(c, rate, &mut self.b);
        let src_frame = self.source_frame(item).unwrap_or(frame);
        for f in c.children().filter(|x| x.has_tag_name("filter")) {
            self.filter(&mut ti, f, rate, frame, src_frame);
        }
        let refs: Vec<String> = children(c, "link").filter_map(|l| child_text(l, "linkclipref")).map(str::to_string).collect();
        Some((ti, refs))
    }

    fn source_frame(&self, item: ItemId) -> Option<(u32, u32)> {
        match &self.b.p.item(item)?.kind {
            ItemKind::Media(m) => m.info.video.as_ref().map(|v| (v.width, v.height)),
            ItemKind::Sequence(s) => Some((s.settings.width, s.settings.height)),
            _ => None,
        }
    }

    fn filter(&mut self, ti: &mut TrackItem, f: Node, rate: FrameRate, frame: (u32, u32), src: (u32, u32)) {
        let Some(e) = child(f, "effect") else { return };
        let enabled = child_bool(f, "enabled").unwrap_or(true);
        let id = child_text(e, "effectid").unwrap_or("");
        let name = child_text(e, "name").unwrap_or(id);
        let params: Vec<Node> = children(e, "parameter").collect();
        let find = |pid: &str| {
            params.iter().copied().find(|p| child_text(*p, "parameterid") == Some(pid) || child_text(*p, "name").is_some_and(|n| n.eq_ignore_ascii_case(pid)))
        };
        let (fw, fh) = (frame.0 as f64, frame.1 as f64);
        let (sw, sh) = (src.0 as f64, src.1 as f64);
        match id {
            "timeremap" => {}
            "basic" => {
                if let Some(p) = find("scale") {
                    set_param(ti, "motion", "scale", read_param(p, rate, |v| v));
                }
                if let Some(p) = find("rotation") {
                    set_param(ti, "motion", "rotation", read_param(p, rate, |v| v));
                }
                if let Some(p) = find("center") {
                    let pr = read_point_param(p, rate, |x, y| Vec2::new(fw / 2.0 + x * fw, fh / 2.0 + y * fh));
                    if pr.is_animated() || pr.value.as_vec2().is_some_and(|v| (v.x - fw / 2.0).abs() > 1e-9 || (v.y - fh / 2.0).abs() > 1e-9) {
                        set_param(ti, "motion", "position", pr);
                    }
                }
                if let Some(p) = find("centerOffset") {
                    let pr = read_point_param(p, rate, |x, y| Vec2::new(sw / 2.0 + x * sw, sh / 2.0 + y * sh));
                    if pr.is_animated() || pr.value.as_vec2().is_some_and(|v| (v.x - sw / 2.0).abs() > 1e-9 || (v.y - sh / 2.0).abs() > 1e-9) {
                        set_param(ti, "motion", "anchor", pr);
                    }
                }
                if let Some(p) = find("antiflicker") {
                    set_param(ti, "motion", "anti_flicker", read_param(p, rate, |v| v));
                }
                if let Some(m) = ti.effect_mut("motion") {
                    m.enabled = enabled;
                }
            }
            "opacity" => {
                if let Some(p) = find("opacity") {
                    set_param(ti, "opacity", "opacity", read_param(p, rate, |v| v));
                }
            }
            "audiolevels" => {
                if let Some(p) = find("level") {
                    set_param(ti, "volume", "level", read_param(p, rate, gain_to_db));
                }
            }
            "audiopan" => {
                if let Some(p) = find("pan") {
                    set_param(ti, "panner", "balance", read_param(p, rate, |v| v * 100.0));
                }
            }
            _ => {
                let def = filmcraft_project::find_effect(id).filter(|d| !d.intrinsic).or_else(|| find_effect_by_name(name).filter(|d| !d.intrinsic));
                let Some(def) = def else {
                    self.report.warn(format!("effect \"{name}\" is not supported and was skipped"));
                    return;
                };
                let mut inst = def.instance();
                inst.enabled = enabled;
                for pd in &def.params {
                    let Some(p) = find(pd.id).or_else(|| find(pd.label)) else { continue };
                    let pr = match &pd.default {
                        ParamValue::Float(_) => read_param(p, rate, |v| v),
                        ParamValue::Vec2(_) => read_point_param(p, rate, Vec2::new),
                        ParamValue::Bool(_) => Param::new(ParamValue::Bool(child_bool(p, "value").unwrap_or(false))),
                        ParamValue::Choice(_) => Param::new(ParamValue::Choice(child_i64(p, "value").unwrap_or(0).max(0) as u32)),
                        ParamValue::Color(_) => Param::new(ParamValue::Color(child(p, "value").map(read_color).unwrap_or([0.0, 0.0, 0.0, 1.0]))),
                        ParamValue::Text(_) => Param::new(ParamValue::Text(child_text(p, "value").unwrap_or("").to_string())),
                        ParamValue::Curve(_) | ParamValue::Path(_) => continue,
                    };
                    inst.params.insert(pd.id.to_string(), pr);
                }
                ti.effects.push(inst);
            }
        }
    }

    fn transition(&mut self, seq: &mut Sequence, kind: TrackKind, ti: usize, c: Node, rate: FrameRate) {
        let start = child_i64(c, "start").unwrap_or(0);
        let end = child_i64(c, "end").unwrap_or(start);
        let (s, e) = (rate.tick_of(start), rate.tick_of(end));
        if e <= s {
            return;
        }
        let eff = child(c, "effect");
        let name = eff.and_then(|x| child_text(x, "name")).or_else(|| eff.and_then(|x| child_text(x, "effectid"))).unwrap_or("Cross Dissolve");
        let effect = transition_effect(name, kind == TrackKind::Audio, self.report);
        let align_s = child_text(c, "alignment").unwrap_or("").to_ascii_lowercase();
        let track = &mut seq.tracks_mut(kind)[ti];
        let (mut from, mut to) = (None, None);
        if align_s != "start-black" {
            from = track.items.iter().filter(|i| i.end() >= s && i.end() <= e).max_by_key(|i| i.end()).map(|i| (i.id, i.end()));
        }
        if align_s != "end-black" {
            to = track.items.iter().filter(|i| i.start >= s && i.start <= e).min_by_key(|i| i.start).map(|i| (i.id, i.start));
        }
        // A cut needs both sides to meet.
        if let (Some((_, fe)), Some((_, ts))) = (from, to)
            && fe != ts
        {
            if align_s == "start-black" || (align_s.is_empty() && ts == s) {
                from = None;
            } else {
                to = None;
            }
        }
        if from.is_none() && to.is_none() {
            self.report.warn(format!("transition \"{name}\" at frame {start} has no adjacent clip; skipped"));
            return;
        }
        let cut = from.map(|f| f.1).or(to.map(|t| t.1)).unwrap_or(s);
        let align = match align_s.as_str() {
            "center" => TransitionAlign::CenterAtCut,
            "start" | "start-black" => TransitionAlign::StartAtCut,
            "end" | "end-black" => TransitionAlign::EndAtCut,
            _ if cut == s => TransitionAlign::StartAtCut,
            _ if cut == e => TransitionAlign::EndAtCut,
            _ => TransitionAlign::CenterAtCut,
        };
        let id = TransitionId(self.b.alloc());
        track.transitions.push(Transition { id, effect, start: s, duration: e - s, from: from.map(|f| f.0), to: to.map(|t| t.0), align, reverse: false });
        track.sort();
    }

    fn resolve_links(&mut self) {
        // Union clip items that reference each other.
        let mut group_of: HashMap<(ItemId, ClipId), u64> = HashMap::new();
        let pending = std::mem::take(&mut self.links);
        for pl in &pending {
            let mut members: Vec<ClipId> = pl.refs.iter().filter_map(|r| self.xml_clip_ids.get(&(pl.seq, r.clone())).copied()).collect();
            members.push(pl.clip);
            members.sort();
            members.dedup();
            if members.len() < 2 {
                continue;
            }
            let existing = members.iter().find_map(|m| group_of.get(&(pl.seq, *m)).copied());
            let g = existing.unwrap_or_else(|| self.b.link_id());
            for m in members {
                group_of.insert((pl.seq, m), g);
            }
            let _ = (pl.kind, pl.track);
        }
        for ((seq, clip), g) in group_of {
            if let Some(s) = self.b.p.sequence_mut(seq)
                && let Some((_, it)) = s.find_item_mut(clip)
            {
                it.link = Some(g);
            }
        }
    }

    /// Put the halves of a stereo clip back together: two linked one-channel clips of the same
    /// two-channel file, at the same times, playing its two channels hard left and hard right,
    /// with the same effects (as our export writes a stereo clip, and Premiere a split one) become
    /// one stereo clip on the left half's track. Levels that differ by a constant become the
    /// clip's Balance. An audio track the right halves leave empty is removed.
    fn merge_stereo_halves(&mut self, seq_id: ItemId) {
        let p = &self.b.p;
        let Some(seq) = p.sequence(seq_id) else { return };
        let stereo_file = |id: ItemId| item_media(p, base_item(p, id)).and_then(|m| m.info.audio()).is_some_and(|a| a.channels == 2);
        let side = |it: &TrackItem| match it.effect("panner").filter(|e| e.enabled).and_then(|e| e.param("balance")) {
            Some(b) if !b.is_animated() => b.value.as_f64().filter(|v| v.abs() == 100.0),
            _ => None,
        };
        // (left track, left clip, right track, right clip, Balance, Volume level)
        let mut merges: Vec<(usize, ClipId, usize, ClipId, f64, Option<Param>)> = Vec::new();
        let mut taken: HashSet<ClipId> = HashSet::new();
        let tracks = seq.tracks(TrackKind::Audio);
        let cands: Vec<(usize, &TrackItem)> = tracks
            .iter()
            .enumerate()
            .flat_map(|(ti, t)| t.items.iter().map(move |c| (ti, c)))
            .filter(|(_, c)| c.source_channels.len() == 1 && c.link.is_some() && stereo_file(c.item) && side(c) == Some(-100.0))
            .collect();
        for (lt, l) in cands {
            let r = tracks.iter().enumerate().filter(|(rt, _)| *rt != lt).flat_map(|(rt, t)| t.items.iter().map(move |c| (rt, c))).find(|(_, r)| {
                !taken.contains(&r.id)
                    && r.link == l.link
                    && r.item == l.item
                    && r.source_channels.len() == 1
                    && r.source_channels != l.source_channels
                    && side(r) == Some(100.0)
                    && (r.start, r.duration, r.source_in, r.speed, r.reverse, r.frame_hold, r.enabled)
                        == (l.start, l.duration, l.source_in, l.speed, l.reverse, l.frame_hold, l.enabled)
                    && r.gain_db == l.gain_db
                    && r.effects
                        .iter()
                        .filter(|e| e.effect != "panner" && e.effect != "volume")
                        .eq(l.effects.iter().filter(|e| e.effect != "panner" && e.effect != "volume"))
            });
            let Some((rt, r)) = r else { continue };
            let level = |c: &TrackItem| c.effect("volume").filter(|e| e.enabled).and_then(|e| e.param("level")).cloned();
            let (ll, rl) = (level(l), level(r));
            let (volume, balance) = if ll == rl {
                (ll, 0.0)
            } else if let Some(d) = level_offset(&ll, &rl) {
                // The louder side is the level, the quieter one's attenuation the Balance.
                let bal = balance_for(db_to_gain(-d.abs()), d > 0.0);
                (Some(if d >= 0.0 { ll } else { rl }.unwrap_or_else(|| Param::new(ParamValue::Float(0.0)))), bal)
            } else {
                continue;
            };
            taken.insert(l.id);
            taken.insert(r.id);
            merges.push((lt, l.id, rt, r.id, balance, volume));
        }
        if merges.is_empty() {
            return;
        }
        let Some(seq) = self.b.p.sequence_mut(seq_id) else { return };
        let mut emptied: Vec<usize> = Vec::new();
        for (_, lid, rt, rid, balance, volume) in merges {
            let Some(rch) = seq.find_item(rid).and_then(|(_, r)| r.source_channels.first().copied()) else { continue };
            if let Some(t) = seq.tracks_mut(TrackKind::Audio).get_mut(rt) {
                t.items.retain(|c| c.id != rid);
                t.transitions.retain(|tr| tr.from != Some(rid) && tr.to != Some(rid));
                if t.items.is_empty() && t.transitions.is_empty() {
                    emptied.push(rt);
                }
            }
            let Some((_, l)) = seq.find_item_mut(lid) else { continue };
            let lch = l.source_channels.first().copied().unwrap_or(0);
            l.source_channels = if (lch, rch) == (0, 1) { Vec::new() } else { vec![lch, rch] };
            set_param(l, "panner", "balance", Param::new(ParamValue::Float(if balance.abs() < 1e-9 { 0.0 } else { balance })));
            set_param(l, "volume", "level", volume.unwrap_or_else(|| Param::new(ParamValue::Float(0.0))));
            // A link group of one is no link.
            if let Some(g) = l.link
                && seq.all_tracks().flat_map(|t| &t.items).filter(|c| c.link == Some(g)).count() < 2
                && let Some((_, l)) = seq.find_item_mut(lid)
            {
                l.link = None;
            }
        }
        emptied.sort_unstable();
        emptied.dedup();
        for t in emptied.into_iter().rev() {
            if seq.audio_tracks.len() > 1 && seq.audio_tracks.get(t).is_some_and(|t| t.items.is_empty() && t.transitions.is_empty()) {
                seq.audio_tracks.remove(t);
            }
        }
    }
}

/// How many dB louder the left half's level is than the right half's, when the two differ by the
/// same amount everywhere: two static levels, or keyframes at the same times and with the same
/// interpolation whose values all differ by one offset (our export folds a Balance in this way).
fn level_offset(l: &Option<Param>, r: &Option<Param>) -> Option<f64> {
    let zero = Param::new(ParamValue::Float(0.0));
    let (l, r) = (l.as_ref().unwrap_or(&zero), r.as_ref().unwrap_or(&zero));
    if !l.is_animated() && !r.is_animated() {
        return Some(l.value.as_f64()? - r.value.as_f64()?);
    }
    if l.keyframes.len() != r.keyframes.len() {
        return None;
    }
    let mut offsets = l.keyframes.iter().zip(&r.keyframes).map(|(a, b)| {
        let same = (a.time, a.interp, a.out_influence, a.in_influence) == (b.time, b.interp, b.out_influence, b.in_influence);
        if same { Some(a.value.as_f64()? - b.value.as_f64()?) } else { None }
    });
    let d = offsets.next()??;
    // The levels went through the XML as gains, so allow for the rounding.
    offsets.all(|o| o.is_some_and(|o| (o - d).abs() < 1e-4)).then_some(d)
}

fn read_color(v: Node) -> [f32; 4] {
    let c = |n: &str, d: f64| (child_f64(v, n).unwrap_or(d) / 255.0) as f32;
    [c("red", 0.0), c("green", 0.0), c("blue", 0.0), c("alpha", 255.0)]
}

fn read_param(p: Node, rate: FrameRate, map: impl Fn(f64) -> f64) -> Param {
    let v = child_f64(p, "value").unwrap_or(0.0);
    let mut pr = Param::new(ParamValue::Float(map(v)));
    for k in children(p, "keyframe") {
        let when = child_i64(k, "when").unwrap_or(0);
        let kv = child_f64(k, "value").unwrap_or(v);
        let mut kf = Keyframe::new(rate.tick_of(when), ParamValue::Float(map(kv)));
        kf.interp = keyframe_interp(k);
        pr.keyframes.push(kf);
    }
    pr.keyframes.sort_by_key(|k| k.time);
    if let Some(k) = pr.keyframes.first() {
        pr.value = k.value.clone();
    }
    pr
}

fn keyframe_interp(k: Node) -> Interpolation {
    match path_text(k, &["interpolation", "name"]).map(str::to_ascii_lowercase).as_deref() {
        Some("hold") => Interpolation::Hold,
        Some("bezier") | Some("fcpcurve") => Interpolation::Bezier,
        _ => Interpolation::Linear,
    }
}

fn read_xy(v: Node) -> (f64, f64) {
    (child_f64(v, "horiz").unwrap_or(0.0), child_f64(v, "vert").unwrap_or(0.0))
}

fn read_point_param(p: Node, rate: FrameRate, map: impl Fn(f64, f64) -> Vec2) -> Param {
    let (x, y) = child(p, "value").map(read_xy).unwrap_or((0.0, 0.0));
    let mut pr = Param::new(ParamValue::Vec2(map(x, y)));
    for k in children(p, "keyframe") {
        let when = child_i64(k, "when").unwrap_or(0);
        let (kx, ky) = child(k, "value").map(read_xy).unwrap_or((x, y));
        let mut kf = Keyframe::new(rate.tick_of(when), ParamValue::Vec2(map(kx, ky)));
        kf.interp = keyframe_interp(k);
        pr.keyframes.push(kf);
    }
    pr.keyframes.sort_by_key(|k| k.time);
    if let Some(k) = pr.keyframes.first() {
        pr.value = k.value.clone();
    }
    pr
}

fn read_markers(n: Node, rate: FrameRate, b: &mut Builder) -> Vec<Marker> {
    let mut out = Vec::new();
    for m in children(n, "marker") {
        let in_f = child_i64(m, "in").unwrap_or(0);
        let out_f = child_i64(m, "out").unwrap_or(-1);
        let start = rate.tick_of(in_f);
        let duration = if out_f > in_f { rate.tick_of(out_f) - start } else { Tick::ZERO };
        let color = child_text(m, "color").and_then(label_from).unwrap_or(Label::Green);
        out.push(Marker {
            id: MarkerId(b.alloc()),
            start,
            duration,
            name: child_text(m, "name").unwrap_or("").to_string(),
            comment: child_text(m, "comment").unwrap_or("").to_string(),
            kind: MarkerKind::Comment,
            color,
        });
    }
    out.sort_by_key(|m| m.start);
    out
}

// ---------------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------------

/// One written clip item, as a `<link>` names it.
#[derive(Clone)]
struct XmlRef {
    id: String,
    kind: TrackKind,
    track: usize,
    index: usize,
    /// The channel group of a stereo clip's halves (`groupindex`).
    group: Option<usize>,
}

/// Which half of a stereo clip a clip item is: its source channel and its side.
#[derive(Clone, Copy)]
struct Half {
    channel: u16,
    right: bool,
}

struct Exp<'a, 'r> {
    p: &'a Project,
    opts: &'a ExportOptions,
    report: &'r mut Report,
    w: XmlWriter,
    file_ids: HashMap<ItemId, String>,
    seq_ids: HashMap<ItemId, String>,
    master_ids: HashMap<ItemId, String>,
    writing: HashSet<ItemId>,
    written_seqs: HashSet<ItemId>,
    next_clip: usize,
}

pub(crate) fn export(p: &Project, seq_id: ItemId, opts: &ExportOptions, report: &mut Report) -> Result<String> {
    p.sequence(seq_id).ok_or(Error::NoSequence(seq_id))?;
    let mut x = Exp {
        p,
        opts,
        report,
        w: XmlWriter::new(Some("xmeml")),
        file_ids: HashMap::new(),
        seq_ids: HashMap::new(),
        master_ids: HashMap::new(),
        writing: HashSet::new(),
        written_seqs: HashSet::new(),
        next_clip: 1,
    };
    let v = if opts.xmeml_version == 5 { "5" } else { "4" };
    x.w.open("xmeml", &[("version", v)]);
    x.write_master_clips(seq_id);
    x.write_sequence(seq_id, true);
    x.w.close();
    Ok(x.w.finish())
}

impl Exp<'_, '_> {
    fn seq_id(&mut self, id: ItemId) -> String {
        let n = self.seq_ids.len() + 1;
        self.seq_ids.entry(id).or_insert_with(|| format!("sequence-{n}")).clone()
    }

    fn write_timecode(&mut self, frame: i64, rate: FrameRate, df: bool) {
        self.w.open("timecode", &[]);
        write_rate(&mut self.w, rate);
        self.w.text("string", format_timecode_frames(frame, rate, df));
        self.w.text("frame", frame);
        self.w.text("displayformat", if df { "DF" } else { "NDF" });
        self.w.close();
    }

    fn write_sequence(&mut self, id: ItemId, top: bool) {
        let sid = self.seq_id(id);
        if self.written_seqs.contains(&id) || self.writing.contains(&id) {
            self.w.empty("sequence", &[("id", &sid)]);
            return;
        }
        let Some((it, seq)) = self.p.item(id).and_then(|it| Some((it, it.as_sequence()?))) else {
            self.w.empty("sequence", &[("id", &sid)]);
            return;
        };
        self.writing.insert(id);
        let rate = seq.settings.frame_rate;
        let df = seq.settings.drop_frame && rate.supports_drop_frame();
        self.w.open("sequence", &[("id", &sid)]);
        self.w.text("name", self.opts.name.clone().filter(|_| top).unwrap_or_else(|| it.name.clone()));
        self.w.text("duration", frames_round(rate, seq.duration()));
        write_rate(&mut self.w, rate);
        self.write_timecode(seq.start_timecode, rate, df);
        self.w.open("labels", &[]);
        self.w.text("label2", it.label.name());
        self.w.close();
        self.w.open("media", &[]);
        let mut link_members: HashMap<u64, Vec<ClipId>> = HashMap::new();
        for t in seq.all_tracks() {
            for c in &t.items {
                if let Some(l) = c.link {
                    link_members.entry(l).or_default().push(c.id);
                }
            }
        }
        // A stereo clip is written as two mono clip items, its left and right source channels, on
        // a pair of XML tracks, as Premiere writes it: one clip item with source track 1 plays in
        // Premiere as channel 1 alone, centred (#2152). So an audio track holding a stereo clip
        // becomes two XML tracks, the second holding the right halves.
        let halves: HashMap<ClipId, [u16; 2]> =
            seq.tracks(TrackKind::Audio).iter().flat_map(|t| &t.items).filter_map(|c| Some((c.id, self.stereo_halves(c)?))).collect();
        let paired: Vec<bool> = seq.tracks(TrackKind::Audio).iter().map(|t| t.items.iter().any(|c| halves.contains_key(&c.id))).collect();
        // Link table: ClipId -> its clip items' (xml id, kind, XML track index, clip index, channel group).
        let mut clip_xml: HashMap<ClipId, Vec<XmlRef>> = HashMap::new();
        for kind in [TrackKind::Video, TrackKind::Audio] {
            let mut xml_track = 1;
            for (ti, t) in seq.tracks(kind).iter().enumerate() {
                let mut right_index = 0;
                for (ci, c) in t.items.iter().enumerate() {
                    let mut refs = vec![XmlRef { id: format!("clipitem-{}", self.next_clip), kind, track: xml_track, index: ci + 1, group: None }];
                    self.next_clip += 1;
                    if halves.contains_key(&c.id) {
                        // The channel group of the pair among the stereo clips of its link group.
                        let group = c
                            .link
                            .and_then(|l| link_members.get(&l))
                            .and_then(|m| m.iter().filter(|id| halves.contains_key(id)).position(|id| *id == c.id))
                            .unwrap_or(0)
                            + 1;
                        right_index += 1;
                        refs[0].group = Some(group);
                        refs.push(XmlRef { id: format!("clipitem-{}", self.next_clip), kind, track: xml_track + 1, index: right_index, group: Some(group) });
                        self.next_clip += 1;
                    }
                    clip_xml.insert(c.id, refs);
                }
                xml_track += if paired.get(ti) == Some(&true) { 2 } else { 1 };
            }
        }
        if seq.all_tracks().any(|t| t.items.iter().any(|i| i.group.is_some())) {
            self.report.info("clip groups have no FCP7 XML equivalent and were not exported");
        }
        for kind in [TrackKind::Video, TrackKind::Audio] {
            let tag = if kind == TrackKind::Video { "video" } else { "audio" };
            self.w.open(tag, &[]);
            self.w.open("format", &[]);
            self.w.open("samplecharacteristics", &[]);
            if kind == TrackKind::Video {
                write_rate(&mut self.w, rate);
                self.w.text("width", seq.settings.width);
                self.w.text("height", seq.settings.height);
                self.w.text("anamorphic", "FALSE");
                self.w.text("pixelaspectratio", if seq.settings.par == (1, 1) { "square" } else { "custom" });
                self.w.text("fielddominance", "none");
            } else {
                self.w.text("depth", 16);
                self.w.text("samplerate", seq.settings.sample_rate);
            }
            self.w.close();
            self.w.close();
            for (ti, t) in seq.tracks(kind).iter().enumerate() {
                let pair = kind == TrackKind::Audio && paired.get(ti) == Some(&true);
                for side in 0..if pair { 2 } else { 1 } {
                    self.w.open("track", &[]);
                    // Clip and transition items in time order; the right track has the right halves
                    // and the transitions between them.
                    let mut order: Vec<(Tick, u8, usize)> =
                        t.items.iter().enumerate().filter(|(_, c)| side == 0 || halves.contains_key(&c.id)).map(|(i, c)| (c.start, 1, i)).collect();
                    order.extend(
                        t.transitions
                            .iter()
                            .enumerate()
                            .filter(|(_, tr)| side == 0 || [tr.from, tr.to].iter().flatten().any(|id| halves.contains_key(id)))
                            .map(|(i, tr)| (tr.start, 0, i)),
                    );
                    order.sort();
                    for (_, k, i) in order {
                        if k == 1 {
                            let Some(c) = t.items.get(i) else { continue };
                            // A stereo clip without a link group still links its two halves.
                            let members = match c.link.and_then(|l| link_members.get(&l)) {
                                Some(m) => m.clone(),
                                None if halves.contains_key(&c.id) => vec![c.id],
                                None => Vec::new(),
                            };
                            let links: Vec<XmlRef> = members.iter().filter_map(|id| clip_xml.get(id)).flatten().cloned().collect();
                            let Some(xid) = clip_xml.get(&c.id).and_then(|r| r.get(side)).map(|r| r.id.clone()) else { continue };
                            let half = halves.get(&c.id).map(|ch| Half { channel: ch[side], right: side == 1 });
                            self.write_clip(c, kind, seq, &xid, &links, half);
                        } else if let Some(tr) = t.transitions.get(i) {
                            if side == 1 && ![tr.from, tr.to].iter().flatten().all(|id| halves.contains_key(id)) {
                                // A crossfade with a mono clip: the right track has only the stereo
                                // side, which fades on its own over its part of the transition.
                                let Some(one) = one_sided(tr, t, &halves) else { continue };
                                self.report
                                    .warn("a crossfade between a stereo and a mono clip fades the stereo clip's right channel over its own side of the cut");
                                self.write_transition(&one, kind, rate, t);
                                continue;
                            }
                            self.write_transition(tr, kind, rate, t);
                        }
                    }
                    self.w.text("enabled", bool_str(t.enabled));
                    self.w.text("locked", bool_str(t.locked));
                    if pair {
                        self.w.text("outputchannelindex", side + 1);
                    }
                    self.w.close();
                }
            }
            self.w.close();
        }
        self.w.close(); // media
        write_markers(&mut self.w, &seq.markers, rate);
        self.w.close(); // sequence
        self.writing.remove(&id);
        self.written_seqs.insert(id);
    }

    fn write_transition(&mut self, tr: &Transition, kind: TrackKind, rate: FrameRate, t: &filmcraft_project::Track) {
        if !on_frame(rate, tr.start) || !on_frame(rate, tr.duration) {
            self.report.warn("sub-frame transition positions were rounded to frames");
        }
        let align = match (tr.from, tr.to, tr.align) {
            (None, Some(_), _) => "start-black",
            (Some(_), None, _) => "end-black",
            (_, _, TransitionAlign::CenterAtCut) => "center",
            (_, _, TransitionAlign::StartAtCut) => "start",
            (_, _, TransitionAlign::EndAtCut) => "end",
        };
        let _ = t;
        self.w.open("transitionitem", &[]);
        write_rate(&mut self.w, rate);
        self.w.text("start", frames_round(rate, tr.start));
        self.w.text("end", frames_round(rate, tr.end()));
        self.w.text("alignment", align);
        self.w.open("effect", &[]);
        let name = transition_name(&tr.effect);
        self.w.text("name", &name);
        self.w.text("effectid", &name);
        self.w.text("effectcategory", tr.effect.def().and_then(|d| d.category.get(1).copied()).unwrap_or("Dissolve"));
        self.w.text("effecttype", "transition");
        self.w.text("mediatype", if kind == TrackKind::Video { "video" } else { "audio" });
        self.w.close();
        self.w.close();
    }

    fn master_id(&mut self, item: ItemId) -> String {
        let n = self.master_ids.len() + 1;
        self.master_ids.entry(item).or_insert_with(|| format!("masterclip-{n}")).clone()
    }

    /// Source-media (master clip) markers live on `<clip>`, not on timeline `<clipitem>`s.
    fn write_master_clips(&mut self, seq_id: ItemId) {
        let mut seen = HashSet::new();
        let mut items = Vec::new();
        self.collect_file_media(seq_id, &mut seen, &mut items);
        for item in items {
            let Some(m) = item_media(self.p, item) else { continue };
            if m.markers.is_empty() {
                continue;
            }
            let rate = m
                .info
                .video
                .as_ref()
                .map(|v| m.interpret.frame_rate.unwrap_or(v.frame_rate))
                .unwrap_or_else(|| self.p.sequence(seq_id).map(|s| s.settings.frame_rate).unwrap_or_default());
            let name = self.p.item(item).map(|i| i.name.clone()).unwrap_or_default();
            let label = self.p.item(item).map(|i| i.label).unwrap_or(Label::Iris);
            let duration = m.info.duration;
            let markers = m.markers.clone();
            let mid = self.master_id(item);
            self.w.open("clip", &[("id", &mid)]);
            self.w.text("name", name);
            self.w.text("duration", frames_round(rate, duration));
            write_rate(&mut self.w, rate);
            self.w.open("labels", &[]);
            self.w.text("label2", label.name());
            self.w.close();
            self.write_file(item, rate);
            write_markers(&mut self.w, &markers, rate);
            self.w.close();
        }
    }

    fn collect_file_media(&self, seq: ItemId, seen: &mut HashSet<ItemId>, out: &mut Vec<ItemId>) {
        if !seen.insert(seq) {
            return;
        }
        let Some(s) = self.p.sequence(seq) else { return };
        for t in s.all_tracks() {
            for c in &t.items {
                let base = base_item(self.p, c.item);
                match self.p.item(base).map(|i| &i.kind) {
                    Some(ItemKind::Sequence(_)) => self.collect_file_media(base, seen, out),
                    Some(ItemKind::Media(_)) if crate::common::generator_of(self.p, base).is_none() && !out.contains(&base) => out.push(base),
                    _ => {}
                }
            }
        }
    }

    /// The source channels (left, right) of a clip written as two mono halves: an audio clip of a
    /// two-channel media file playing both channels.
    fn stereo_halves(&self, c: &TrackItem) -> Option<[u16; 2]> {
        let base = base_item(self.p, c.item);
        if crate::common::generator_of(self.p, base).is_some() || item_media(self.p, base)?.info.audio()?.channels != 2 {
            return None;
        }
        match c.source_channels.as_slice() {
            [] => Some([0, 1]),
            &[l, r] if l != r && l < 2 && r < 2 => Some([l, r]),
            _ => None,
        }
    }

    fn write_clip(&mut self, c: &TrackItem, kind: TrackKind, seq: &Sequence, xid: &str, links: &[XmlRef], half: Option<Half>) {
        let rate = seq.settings.frame_rate;
        if !on_frame(rate, c.start) || !on_frame(rate, c.duration) {
            self.report.warn("sub-frame clip positions were rounded to frames");
        }
        if let Some(clip) = crate::common::uncarried_clip(self.p, c, rate) {
            self.report
                .warn(format!("{clip} has no Final Cut Pro 7 XML equivalent: it is written as a clip item without media and is not read back on import"));
        }
        let base = base_item(self.p, c.item);
        let generator = crate::common::generator_of(self.p, base).cloned();
        let is_seq = matches!(self.p.item(base).map(|i| &i.kind), Some(ItemKind::Sequence(_)));
        let tag = if generator.is_some() { "generatoritem" } else { "clipitem" };
        let hold = c.frame_hold;
        let src_in = hold.unwrap_or(c.source_in);
        let src_out = if hold.is_some() { src_in + c.duration } else { c.source_out() };
        let (start_f, end_f, in_f) = (frames_round(rate, c.start), frames_round(rate, c.end()), frames_round(rate, src_in));
        // At normal speed (and on a freeze frame) the source span is the record span. Rounded on its own
        // it can come out a frame shorter or longer than `end - start` (#340), so derive it from them.
        let out_f = if hold.is_some() || c.speed.abs() == 1.0 { in_f.saturating_add(end_f.saturating_sub(start_f)) } else { frames_round(rate, src_out) };
        self.w.open(tag, &[("id", xid)]);
        if generator.is_none() && !is_seq {
            let mid = self.master_id(base);
            self.w.text("masterclipid", mid);
        }
        self.w.text("name", &c.name);
        self.w.text("enabled", bool_str(c.enabled));
        let media_dur = self.p.item(base).map(|i| i.duration()).unwrap_or(Tick::ZERO).max(src_out);
        self.w.text("duration", frames_round(rate, media_dur).max(out_f));
        write_rate(&mut self.w, rate);
        self.w.text("start", start_f);
        self.w.text("end", end_f);
        self.w.text("in", in_f);
        self.w.text("out", out_f);
        self.w.text("pproTicksIn", src_in.0);
        self.w.text("pproTicksOut", src_out.0);
        if let Some(g) = &generator {
            self.write_generator(g);
        } else if is_seq {
            self.write_sequence(base, false);
        } else {
            self.write_file(base, rate);
        }
        if kind == TrackKind::Audio {
            self.w.open("sourcetrack", &[]);
            self.w.text("mediatype", "audio");
            let ch = half.map(|h| h.channel).or_else(|| c.source_channels.first().copied());
            self.w.text("trackindex", ch.map_or(1, |ch| u32::from(ch).saturating_add(1)));
            self.w.close();
        }
        if let Some(blend) = param(c, "opacity", "blend").and_then(|p| p.value.as_f64()).map(|v| v as u32).filter(|v| *v != 0) {
            match BLEND_XML.iter().find(|(_, v)| *v == blend) {
                Some((n, _)) => self.w.text("compositemode", n),
                None => self.report.warn("blend mode not representable in FCP7 XML; Normal used"),
            }
        }
        self.write_filters(c, kind, seq, base, half);
        for l in links {
            self.w.open("link", &[]);
            self.w.text("linkclipref", &l.id);
            self.w.text("mediatype", if l.kind == TrackKind::Video { "video" } else { "audio" });
            self.w.text("trackindex", l.track);
            self.w.text("clipindex", l.index);
            if let Some(g) = l.group {
                self.w.text("groupindex", g);
            }
            self.w.close();
        }
        self.w.open("labels", &[]);
        self.w.text("label2", c.label.name());
        self.w.close();
        // The markers of a stereo clip go on its left half only, so they are not doubled.
        if !half.is_some_and(|h| h.right) {
            write_markers(&mut self.w, &c.markers, rate);
        }
        self.w.close();
    }

    fn write_generator(&mut self, g: &Generator) {
        self.w.open("effect", &[]);
        match g {
            Generator::BlackVideo => {
                self.w.text("name", "Slug");
                self.w.text("effectid", "slug");
            }
            Generator::ColorMatte { color } => {
                self.w.text("name", "Color");
                self.w.text("effectid", "Color");
                self.w.text("effectcategory", "Matte");
                self.w.text("effecttype", "generator");
                self.w.text("mediatype", "video");
                self.w.open("parameter", &[]);
                self.w.text("parameterid", "fillcolor");
                self.w.text("name", "Color");
                write_color(&mut self.w, *color);
                self.w.close();
                self.w.close();
                return;
            }
            other => {
                self.report.info(format!("generator \"{}\" is written with a FilmCraft effect id", other.label()));
                self.w.text("name", other.label());
                self.w.text("effectid", format!("filmcraft:{}", serde_json::to_string(other).unwrap_or_default()));
            }
        }
        self.w.text("effectcategory", "Matte");
        self.w.text("effecttype", "generator");
        self.w.text("mediatype", "video");
        self.w.close();
    }

    fn write_file(&mut self, item: ItemId, seq_rate: FrameRate) {
        if let Some(fid) = self.file_ids.get(&item) {
            let fid = fid.clone();
            self.w.empty("file", &[("id", &fid)]);
            return;
        }
        let fid = format!("file-{}", self.file_ids.len() + 1);
        self.file_ids.insert(item, fid.clone());
        let Some(m) = item_media(self.p, item) else {
            self.w.empty("file", &[("id", &fid)]);
            return;
        };
        let name = self.p.item(item).map(|i| i.name.clone()).unwrap_or_default();
        let frate = m.info.video.as_ref().map(|v| m.interpret.frame_rate.unwrap_or(v.frame_rate)).unwrap_or(seq_rate);
        self.w.open("file", &[("id", &fid)]);
        let path = item_path(self.p, item).unwrap_or("");
        self.w.text("name", if name.is_empty() { file_name(path).to_string() } else { name });
        if crate::common::is_absolute(path) {
            self.w.text("pathurl", path_to_file_url(path, true));
        } else {
            self.w.text("pathurl", path);
        }
        write_rate(&mut self.w, frate);
        self.w.text("duration", frames_round(frate, m.info.duration));
        let tc = m.info.start_timecode.unwrap_or(0);
        self.write_timecode(tc, frate, false);
        self.w.open("media", &[]);
        if let Some(v) = &m.info.video {
            self.w.open("video", &[]);
            self.w.open("samplecharacteristics", &[]);
            write_rate(&mut self.w, frate);
            self.w.text("width", v.width);
            self.w.text("height", v.height);
            self.w.text("anamorphic", "FALSE");
            self.w.text("pixelaspectratio", if v.par == (1, 1) { "square" } else { "custom" });
            self.w.text("fielddominance", "none");
            self.w.close();
            self.w.close();
        }
        if let Some(a) = m.info.audio() {
            self.w.open("audio", &[]);
            self.w.open("samplecharacteristics", &[]);
            self.w.text("depth", a.bits_per_sample.unwrap_or(16));
            self.w.text("samplerate", a.sample_rate);
            self.w.close();
            self.w.text("channelcount", a.channels);
            self.w.close();
        }
        self.w.close();
        self.w.close();
    }

    fn write_filters(&mut self, c: &TrackItem, kind: TrackKind, seq: &Sequence, base: ItemId, half: Option<Half>) {
        let rate = seq.settings.frame_rate;
        let (fw, fh) = (seq.settings.width as f64, seq.settings.height as f64);
        let (sw, sh) = match self.p.item(base).map(|i| &i.kind) {
            Some(ItemKind::Media(m)) => m.info.video.as_ref().map(|v| (v.width as f64, v.height as f64)).unwrap_or((fw, fh)),
            Some(ItemKind::Sequence(s)) => (s.settings.width as f64, s.settings.height as f64),
            _ => (fw, fh),
        };
        if kind == TrackKind::Video {
            let motion_changed = ["position", "scale", "scale_width", "rotation", "anchor", "anti_flicker"].iter().any(|p| param_modified(c, "motion", p));
            if motion_changed {
                if param_modified(c, "motion", "scale_width") || param(c, "motion", "uniform_scale").and_then(|p| p.value.as_bool()) == Some(false) {
                    self.report.warn("non-uniform scale is not supported by FCP7 Basic Motion; uniform scale written");
                }
                let enabled = c.effect("motion").is_none_or(|e| e.enabled);
                self.open_effect("Basic Motion", "basic", "motion", "motion", "video", enabled);
                self.float_param(c, "motion", "scale", "Scale", 0.0, 1000.0, rate, |v| v, "scale");
                self.float_param(c, "motion", "rotation", "Rotation", -8640.0, 8640.0, rate, |v| v, "rotation");
                self.point_param(c, "motion", "position", "Center", rate, "center", (fw, fh), |v| ((v.x - fw / 2.0) / fw, (v.y - fh / 2.0) / fh));
                self.point_param(c, "motion", "anchor", "Anchor Point", rate, "centerOffset", (sw, sh), |v| ((v.x - sw / 2.0) / sw, (v.y - sh / 2.0) / sh));
                if param_modified(c, "motion", "anti_flicker") {
                    self.float_param(c, "motion", "anti_flicker", "Anti-flicker Filter", 0.0, 1.0, rate, |v| v, "antiflicker");
                }
                self.close_effect();
            }
            if param_modified(c, "opacity", "opacity") {
                self.open_effect("Opacity", "opacity", "motion", "motion", "video", true);
                self.float_param(c, "opacity", "opacity", "opacity", 0.0, 100.0, rate, |v| v, "opacity");
                self.close_effect();
            }
            if c.speed != 1.0 || c.reverse || c.frame_hold.is_some() {
                if param(c, "time_remap", "speed").is_some_and(|p| p.is_animated()) {
                    self.report.warn("speed ramps (time remapping keyframes) are not exported to FCP7 XML");
                }
                self.open_effect("Time Remap", "timeremap", "motion", "motion", "video", true);
                let sp = if c.frame_hold.is_some() { 0.0 } else { c.speed * 100.0 };
                self.simple_param("variablespeed", "variablespeed", "0");
                self.simple_param("speed", "speed", &num(sp));
                self.simple_param("reverse", "reverse", bool_str(c.reverse));
                self.simple_param("frameblending", "frameblending", "FALSE");
                self.close_effect();
            }
        } else if let Some(h) = half {
            // A half of a stereo clip plays hard on its side; the clip's Balance is folded into the
            // halves' levels, the attenuation each side gets from it.
            if c.gain_db != 0.0 && !h.right {
                self.report.info("clip gain was folded into Audio Levels");
            }
            let pan = c.effect("panner").filter(|e| e.enabled).and_then(|e| e.param("balance"));
            if pan.is_some_and(|p| p.is_animated()) && !h.right {
                self.report.warn("animated Balance on a stereo clip is not supported by FCP7 XML; its first value was written");
            }
            let b = pan.and_then(|p| p.keyframes.first().map(|k| &k.value).unwrap_or(&p.value).as_f64()).unwrap_or(0.0);
            let (gl, gr) = balance_gains(b);
            let g = c.gain_db + gain_to_db(if h.right { gr } else { gl });
            if param_modified(c, "volume", "level") || g != 0.0 {
                self.open_effect("Audio Levels", "audiolevels", "audiolevels", "audiolevels", "audio", true);
                self.float_param(c, "volume", "level", "Level", 0.0, 3.98109, rate, move |v| db_to_gain(v + g), "level");
                self.close_effect();
            }
            self.open_effect("Audio Pan", "audiopan", "audiopan", "audiopan", "audio", true);
            self.w.open("parameter", &[]);
            self.w.text("parameterid", "pan");
            self.w.text("name", "Pan");
            self.w.text("valuemin", "-1");
            self.w.text("valuemax", "1");
            self.w.text("value", if h.right { "1" } else { "-1" });
            self.w.close();
            self.close_effect();
            if param_modified(c, "channel_volume", "left") || param_modified(c, "channel_volume", "right") {
                self.report.warn("Channel Volume is not supported by FCP7 XML and was not exported");
            }
        } else {
            if c.gain_db != 0.0 {
                self.report.info("clip gain was folded into Audio Levels");
            }
            if param_modified(c, "volume", "level") || c.gain_db != 0.0 {
                let g = c.gain_db;
                self.open_effect("Audio Levels", "audiolevels", "audiolevels", "audiolevels", "audio", true);
                self.float_param(c, "volume", "level", "Level", 0.0, 3.98109, rate, move |v| db_to_gain(v + g), "level");
                self.close_effect();
            }
            // A one-channel clip says where it is panned even when centred: without Audio Pan the
            // importer pans the halves of a split stereo file to their sides (#371).
            if param_modified(c, "panner", "balance") || c.source_channels.len() == 1 {
                self.open_effect("Audio Pan", "audiopan", "audiopan", "audiopan", "audio", true);
                self.float_param(c, "panner", "balance", "Pan", -1.0, 1.0, rate, |v| v / 100.0, "pan");
                self.close_effect();
            }
            if param_modified(c, "channel_volume", "left") || param_modified(c, "channel_volume", "right") {
                self.report.warn("Channel Volume is not supported by FCP7 XML and was not exported");
            }
        }
        for e in standard_effects(c) {
            self.generic_effect(e, kind, rate);
        }
    }

    fn open_effect(&mut self, name: &str, id: &str, category: &str, etype: &str, media: &str, enabled: bool) {
        self.w.open("filter", &[]);
        if !enabled {
            self.w.text("enabled", "FALSE");
        }
        self.w.open("effect", &[]);
        self.w.text("name", name);
        self.w.text("effectid", id);
        self.w.text("effectcategory", category);
        self.w.text("effecttype", etype);
        self.w.text("mediatype", media);
    }

    fn close_effect(&mut self) {
        self.w.close();
        self.w.close();
    }

    fn simple_param(&mut self, id: &str, name: &str, value: &str) {
        self.w.open("parameter", &[]);
        self.w.text("parameterid", id);
        self.w.text("name", name);
        self.w.text("value", value);
        self.w.close();
    }

    #[allow(clippy::too_many_arguments)]
    fn float_param(
        &mut self,
        c: &TrackItem,
        effect: &str,
        pid: &str,
        label: &str,
        min: f64,
        max: f64,
        rate: FrameRate,
        map: impl Fn(f64) -> f64,
        xml_id: &str,
    ) {
        let default = find_effect(effect).and_then(|d| d.param(pid)).and_then(|p| p.default.as_f64()).unwrap_or(0.0);
        let p = param(c, effect, pid).cloned().unwrap_or_else(|| Param::new(ParamValue::Float(default)));
        self.w.open("parameter", &[]);
        self.w.text("parameterid", xml_id);
        self.w.text("name", label);
        self.w.text("valuemin", num(min));
        self.w.text("valuemax", num(max));
        self.w.text("value", num(map(p.value.as_f64().unwrap_or(default))));
        self.keyframes(&p, rate, |v| v.as_f64().map(|x| num(map(x))), None::<fn(&mut XmlWriter, &ParamValue)>);
        self.w.close();
    }

    #[allow(clippy::too_many_arguments)]
    fn point_param(
        &mut self,
        c: &TrackItem,
        effect: &str,
        pid: &str,
        label: &str,
        rate: FrameRate,
        xml_id: &str,
        size: (f64, f64),
        map: impl Fn(Vec2) -> (f64, f64),
    ) {
        let centre = Vec2::new(size.0 / 2.0, size.1 / 2.0);
        let fix = |v: Vec2| Vec2::new(if v.x.is_nan() { centre.x } else { v.x }, if v.y.is_nan() { centre.y } else { v.y });
        let p = param(c, effect, pid).cloned().unwrap_or_else(|| Param::new(ParamValue::Vec2(centre)));
        self.w.open("parameter", &[]);
        self.w.text("parameterid", xml_id);
        self.w.text("name", label);
        let (x, y) = map(fix(p.value.as_vec2().unwrap_or(centre)));
        self.w.open("value", &[]);
        self.w.text("horiz", num(x));
        self.w.text("vert", num(y));
        self.w.close();
        for k in &p.keyframes {
            let (kx, ky) = map(fix(k.value.as_vec2().unwrap_or(centre)));
            self.w.open("keyframe", &[]);
            self.w.text("when", frames_round(rate, k.time));
            self.w.open("value", &[]);
            self.w.text("horiz", num(kx));
            self.w.text("vert", num(ky));
            self.w.close();
            self.write_interp(k.interp);
            self.w.close();
        }
        self.w.close();
    }

    fn write_interp(&mut self, i: Interpolation) {
        match i {
            Interpolation::Linear => {}
            Interpolation::Hold => {
                self.w.open("interpolation", &[]);
                self.w.text("name", "hold");
                self.w.close();
            }
            _ => {
                self.report.info("Bezier keyframe easing is written as FCP curve keyframes (handles not preserved)");
                self.w.open("interpolation", &[]);
                self.w.text("name", "FCPCurve");
                self.w.close();
            }
        }
    }

    fn keyframes(&mut self, p: &Param, rate: FrameRate, val: impl Fn(&ParamValue) -> Option<String>, _custom: Option<fn(&mut XmlWriter, &ParamValue)>) {
        for k in &p.keyframes {
            let Some(v) = val(&k.value) else { continue };
            if !on_frame(rate, k.time) {
                self.report.info("sub-frame keyframes were rounded to frames");
            }
            self.w.open("keyframe", &[]);
            self.w.text("when", frames_round(rate, k.time));
            self.w.text("value", v);
            self.write_interp(k.interp);
            self.w.close();
        }
    }

    fn generic_effect(&mut self, e: &EffectInstance, kind: TrackKind, rate: FrameRate) {
        let Some(def) = e.def() else {
            self.report.warn(format!("unknown effect \"{}\" was not exported", e.effect));
            return;
        };
        let media = if kind == TrackKind::Video { "video" } else { "audio" };
        self.open_effect(def.name, def.id, def.category.get(1).copied().unwrap_or("Filter"), "filter", media, e.enabled);
        for pd in &def.params {
            let Some(p) = e.params.get(pd.id) else { continue };
            match &p.value {
                ParamValue::Float(v) => {
                    self.w.open("parameter", &[]);
                    self.w.text("parameterid", pd.id);
                    self.w.text("name", pd.label);
                    self.w.text("value", num(*v));
                    self.keyframes(p, rate, |v| v.as_f64().map(num), None::<fn(&mut XmlWriter, &ParamValue)>);
                    self.w.close();
                }
                ParamValue::Vec2(v) => {
                    self.w.open("parameter", &[]);
                    self.w.text("parameterid", pd.id);
                    self.w.text("name", pd.label);
                    self.w.open("value", &[]);
                    self.w.text("horiz", num(if v.x.is_nan() { 0.0 } else { v.x }));
                    self.w.text("vert", num(if v.y.is_nan() { 0.0 } else { v.y }));
                    self.w.close();
                    self.w.close();
                    if p.is_animated() {
                        self.report.warn(format!("point keyframes of \"{}\" were not exported", def.name));
                    }
                }
                ParamValue::Bool(b) => self.simple_param(pd.id, pd.label, bool_str(*b)),
                ParamValue::Choice(c) => self.simple_param(pd.id, pd.label, &c.to_string()),
                ParamValue::Text(t) => self.simple_param(pd.id, pd.label, t),
                ParamValue::Color(c) => {
                    self.w.open("parameter", &[]);
                    self.w.text("parameterid", pd.id);
                    self.w.text("name", pd.label);
                    write_color(&mut self.w, *c);
                    self.w.close();
                }
                ParamValue::Curve(_) | ParamValue::Path(_) => {
                    self.report.warn(format!("curve parameters of \"{}\" are not supported by FCP7 XML", def.name));
                }
            }
        }
        self.close_effect();
    }
}

/// The stereo side of a transition between a stereo clip and a mono one, as a fade to or from
/// nothing over the part of the transition on that side of the cut.
fn one_sided(tr: &Transition, t: &filmcraft_project::Track, halves: &HashMap<ClipId, [u16; 2]>) -> Option<Transition> {
    let clip = |id: Option<ClipId>| id.filter(|id| halves.contains_key(id)).and_then(|id| t.items.iter().find(|c| c.id == id));
    let (start, end, from, to) = match (clip(tr.from), clip(tr.to)) {
        (Some(c), None) => (tr.start, tr.end().min(c.end()), tr.from, None),
        (None, Some(c)) => (tr.start.max(c.start), tr.end(), None, tr.to),
        _ => return None,
    };
    (end > start).then(|| Transition { start, duration: end - start, from, to, ..tr.clone() })
}

fn write_color(w: &mut XmlWriter, c: [f32; 4]) {
    let b = |v: f32| ((v.clamp(0.0, 1.0) * 255.0).round() as i64).to_string();
    w.open("value", &[]);
    w.text("alpha", b(c[3]));
    w.text("red", b(c[0]));
    w.text("green", b(c[1]));
    w.text("blue", b(c[2]));
    w.close();
}

fn write_markers(w: &mut XmlWriter, markers: &[Marker], rate: FrameRate) {
    for m in markers {
        w.open("marker", &[]);
        w.text("comment", &m.comment);
        w.text("name", &m.name);
        let i = frames_round(rate, m.start);
        w.text("in", i);
        w.text("out", if m.duration > Tick::ZERO { frames_round(rate, m.start + m.duration) } else { -1 });
        w.text("color", m.color.name());
        w.close();
    }
}
