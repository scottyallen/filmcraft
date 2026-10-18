//! Dissolves (the classic Cross Dissolve / Dip / Film / Additive family stays in `legacy.rs`,
//! whose output the golden images pin).

use super::{Tx, add_light, bell, blurred, fbm, hash, luma, mid_mix, mix, paint, scale4, smoothstep};
use rayon::prelude::*;

use crate::image::Image;

/// Perceptual (≈ gamma 2.2) luma of a premultiplied linear pixel, 0..1.
pub(crate) fn perceptual_luma(c: [f32; 4]) -> f32 {
    luma(c).clamp(0.0, 1.0).powf(1.0 / 2.2)
}

/// Cross Dissolve as Premiere Pro renders it (measured against its renders, docs/gaps.md G1): the
/// two frames are mixed in sRGB-encoded (display) values, not in linear light, and the weight of B
/// follows the ease 3p² − 2p³ (Premiere's default Bezier ease) rather than `p`. A mid-dissolve of
/// a bright and a dark frame is therefore darker than a linear-light mix. Alpha mixes linearly;
/// colour is encoded unpremultiplied and re-premultiplied, so a side that is transparent (a
/// dissolve from or to nothing) fades the other side's opacity, as before.
pub(crate) fn cross_dissolve(a: &Image, b: &Image, p: f32) -> Image {
    let w = smoothstep(0.0, 1.0, p);
    let mut out = Image::new(a.w, a.h);
    out.px.par_chunks_mut(4).zip(a.px.par_chunks(4).zip(b.px.par_chunks(4))).for_each(|(o, (x, y))| {
        let (Ok(x), Ok(y)) = (<[f32; 4]>::try_from(x), <[f32; 4]>::try_from(y)) else { return };
        o.copy_from_slice(&encoded_mix(x, y, w));
    });
    out
}

/// Premultiplied linear `a` and `b` mixed `w` of the way in sRGB-encoded values.
pub(crate) fn encoded_mix(a: [f32; 4], b: [f32; 4], w: f32) -> [f32; 4] {
    let alpha = a[3] + (b[3] - a[3]) * w;
    if alpha <= 0.0 || !alpha.is_finite() {
        return [0.0; 4];
    }
    // encoded straight colour, premultiplied again so a transparent side weighs nothing
    let enc = |c: [f32; 4], k: usize| if c[3] > 0.0 { encode(c[k] / c[3]) * c[3] } else { 0.0 };
    let mut out = [0.0, 0.0, 0.0, alpha];
    for (k, v) in out.iter_mut().take(3).enumerate() {
        let e = enc(a, k) + (enc(b, k) - enc(a, k)) * w;
        *v = decode(e / alpha) * alpha;
    }
    out
}

/// The sRGB curve, extended to negative values by symmetry.
fn encode(v: f32) -> f32 {
    filmcraft_color::linear_to_srgb(v.abs()).copysign(v)
}

fn decode(v: f32) -> f32 {
    filmcraft_color::srgb_to_linear(v.abs()).copysign(v)
}

pub(super) fn apply(id: &str, t: &Tx) -> Option<Image> {
    let p = t.p;
    Some(match id {
        "cross_dissolve" => cross_dissolve(t.a, t.b, p),
        "blur_dissolve" => {
            let sigma = t.px("blur") * 0.5;
            let k = smoothstep(0.25, 0.75, p);
            let sa = sigma * smoothstep(0.0, 0.6, p);
            let sb = sigma * (1.0 - smoothstep(0.4, 1.0, p));
            let ba = if k < 1.0 { blurred(t.a, sa) } else { t.a.clone() };
            let bb = if k > 0.0 { blurred(t.b, sb) } else { t.b.clone() };
            ba.lerp(&bb, k)
        }
        "burn_alpha" => {
            let col = t.color("burn_color");
            let soft = t.frac("softness").clamp(0.01, 1.0) * 0.3;
            let seed = t.seed();
            let scale = (220.0 * t.unit()).max(1.0);
            let thr = -soft + p * (1.0 + 2.0 * soft);
            let fade = (p * 10.0).min((1.0 - p) * 10.0).min(1.0);
            paint(t.w, t.h, |x, y| {
                let a = t.pa(x, y);
                // burn order: dark areas of A and the noise go first
                let g = 0.55 * fbm(x / scale, y / scale, seed) + 0.45 * (1.0 - perceptual_luma(a));
                let d = g - thr; // < 0 burnt through
                let k = 1.0 - smoothstep(-soft * 0.3, soft * 0.3, d);
                let base = mix(a, t.pb(x, y), k);
                let edge = (-(d / (soft * 0.6)).powi(2)).exp() * fade;
                let charred = if d > 0.0 { 1.0 - 0.6 * (-(d / soft).powi(2)).exp() * fade } else { 1.0 };
                add_light([base[0] * charred, base[1] * charred, base[2] * charred, base[3]], col, edge * 2.0)
            })
        }
        "dip_to_color" => {
            let col = t.color("color");
            let hold = t.frac("hold").clamp(0.0, 0.9);
            let half = (1.0 - hold) / 2.0;
            paint(t.w, t.h, |x, y| {
                if p < half {
                    mix(t.pa(x, y), col, p / half)
                } else if p > 1.0 - half {
                    mix(col, t.pb(x, y), (p - (1.0 - half)) / half)
                } else {
                    col
                }
            })
        }
        // a jump cut smoothed by morphing: both frames are warped along the dense optical flow
        // between them and blended, so the subject moves from its place in the outgoing shot to
        // its place in the incoming one instead of showing twice; where the frames do not
        // correspond (no consistent motion) it is the cross dissolve
        "morph_cut" => crate::flow::interpolate(t.a, t.b, p),
        "luma_fade" => {
            let soft = t.frac("softness").clamp(0.005, 1.0) * 0.5;
            let from_b = t.choice("source") == 1;
            let inv = t.flag("invert");
            let thr = -soft + p * (1.0 + 2.0 * soft);
            paint(t.w, t.h, |x, y| {
                let (a, b) = (t.pa(x, y), t.pb(x, y));
                let l = perceptual_luma(if from_b { b } else { a });
                let g = if inv { l } else { 1.0 - l };
                mix(a, b, smoothstep(-soft, soft, thr - g))
            })
        }
        "mosaic_transition" => {
            let max = t.px("block_size").max(1.0);
            let bs = 1.0 + (max - 1.0) * bell(p);
            let k = mid_mix(p);
            let per_block = t.flag("dissolve");
            paint(t.w, t.h, |x, y| {
                if bs < 1.5 {
                    return mix(t.pa(x, y), t.pb(x, y), k);
                }
                let (bx, by) = ((x / bs).floor(), (y / bs).floor());
                let (sx, sy) = ((bx + 0.5) * bs, (by + 0.5) * bs);
                let kk = if per_block {
                    let r = hash(bx as i32, by as i32, 7);
                    smoothstep(r - 0.15, r + 0.15, k * 1.3 - 0.15)
                } else {
                    k
                };
                mix(t.sac(sx, sy), t.sbc(sx, sy), kk)
            })
        }
        "non_additive_dissolve" => {
            let (ka, kb) = ((2.0 * (1.0 - p)).min(1.0), (2.0 * p).min(1.0));
            paint(t.w, t.h, |x, y| {
                let a = scale4(t.pa(x, y), ka);
                let b = scale4(t.pb(x, y), kb);
                if luma(a) >= luma(b) { a } else { b }
            })
        }
        _ => return None,
    })
}
