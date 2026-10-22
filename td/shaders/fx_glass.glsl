// fx 'glass' (frosted glass): the room seen through a pane of real-looking frosted glass.
//  * the frost is a per-pixel randomly rotated disc blur (sampled from the camera's mip chain, so it is smooth and grainy like etched glass, never noisy),
//  * the frost thickness varies slowly across the pane; thin patches show the room almost sharply,
//  * condensation droplets: each is a tiny clear lens that shows the room upside-down with a bright rim and a dark shadow edge,
//  * a faint refraction field, chromatic dispersion, a drifting specular glint and an edge-lit cool tint.
// Reactive: hand speed WIPES the frost clear (thinner frost, sharper room), strengthens dispersion and sweeps the glint faster.
// Inputs: 0 camera (mip-mapped)   Vectors: uR = (time, speed, openness, twist 0..1)   uC = (box centre uv.xy, 0, 0)
uniform vec4 uR;
uniform vec4 uC;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 h22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { return 0.5 * vn(p) + 0.25 * vn(p * 2.03 + 7.1) + 0.125 * vn(p * 4.1 + 3.3); }
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }     // interleaved gradient noise

// one droplet layer: returns (lens offset in uv, rim, shade); cell grid in pixels, one drop per cell
vec4 droplets(vec2 px, float cell, float seed, float t) {
    vec2 g = floor(px / cell), f = fract(px / cell);
    vec2 r = h22(g + seed);
    vec2 c = 0.3 + 0.4 * r;                                   // drop centre inside the cell
    float rad = (0.12 + 0.2 * h21(g + seed + 9.0)) * step(0.35, h21(g + seed + 3.0));
    vec2 d = (f - c);
    float dist = length(d) / max(rad, 1e-3);
    float inside = 1.0 - smoothstep(0.92, 1.0, dist);
    vec2 lens = -d / max(rad, 1e-3) * inside;                 // refraction flips the image inside the drop
    float rim = smoothstep(0.65, 0.95, dist) * inside;
    float hi = pow(max(0.0, 1.0 - length(d - vec2(-0.35, 0.35) * rad) / (0.35 * rad)), 2.0) * inside;   // specular dot
    return vec4(lens * rad * cell, rim, hi) * step(0.001, rad);
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    float time = uR.x, speed = uR.y, twist = uR.w;
    vec2 asp = vec2(res.x / res.y, 1.0);

    // frost thickness map (slow, large-scale) and wiping by hand speed
    float thick = smoothstep(0.25, 0.8, fbm(uv * asp * 2.2 + vec2(0.0, time * 0.01)));
    float wipe = smoothstep(0.0, 0.8, speed);
    float R = mix(9.0, 26.0, thick) * (1.0 - 0.7 * wipe);                       // frost radius (px)

    // droplets (two sizes) -> local clear lenses
    vec4 d1 = droplets(px + vec2(0.0, time * 1.2), 46.0, 1.0, time);
    vec4 d2 = droplets(px * 1.0 + vec2(17.0, time * 0.7), 21.0, 5.0, time);
    float dropMask = clamp(max(d1.z, d2.z) * 0.0 + step(0.0001, length(d1.xy)) + step(0.0001, length(d2.xy)), 0.0, 1.0);
    vec2 lens = (d1.xy + d2.xy) / res;
    R *= 1.0 - 0.9 * dropMask;                                                   // drops are clear

    // slow refraction waves through the pane
    float e = 0.012;
    vec2 pw = uv * asp * (2.0 + 1.5 * twist) + vec2(time * 0.03, -time * 0.02);
    float n0 = fbm(pw);
    vec2 grad = vec2(fbm(pw + vec2(e, 0.0)) - n0, fbm(pw + vec2(0.0, e)) - n0) / e;
    vec2 off = grad * (0.004 + 0.014 * speed) + lens;

    // frosted disc blur from the mip chain; every pixel rotates the disc differently (that IS the grain of etched glass)
    const int N = 9;
    float rot = ign(px) * 6.2831853;
    float lod = clamp(log2(R * 0.55 + 1.0), 0.0, 5.0);
    vec3 acc = vec3(0.0);
    float disp = (0.9 + 2.2 * speed) / res.y;                                    // dispersion in uv
    for (int i = 0; i < N; i++) {
        float a = float(i) * 2.39996323 + rot, rr = sqrt((float(i) + 0.5) / float(N)) * R;
        vec2 d = vec2(cos(a), sin(a)) * rr / res;
        vec2 o = off + d;
        acc.r += textureLod(sTD2DInputs[0], uv + o * (1.0 + 18.0 * disp), lod).r;
        acc.g += textureLod(sTD2DInputs[0], uv + o, lod).g;
        acc.b += textureLod(sTD2DInputs[0], uv + o * (1.0 - 18.0 * disp), lod).b;
    }
    vec3 col = acc / float(N);

    // frost sparkle: tiny bright grains where the etched surface catches light
    float sp = pow(h21(floor(px * 0.75) + floor(time * 6.0)), 40.0) * (1.0 - dropMask) * 0.6;
    col += sp * vec3(0.8, 0.9, 1.0) * (0.4 + 0.6 * thick);

    // droplet shading: bright rim, dark outer shadow, specular dot
    float rim = max(d1.z, d2.z), spec = max(d1.w, d2.w);
    col = col * (1.0 - 0.25 * rim) + vec3(0.85, 0.95, 1.0) * (0.18 * rim + 0.9 * spec);

    // drifting specular glint across the pane
    float gl = dot(uv * asp, normalize(vec2(1.0, 0.55))) - fract(time * (0.045 + 0.1 * speed)) * 2.6 + 0.4;
    col += vec3(0.75, 0.9, 1.0) * exp(-gl * gl * 55.0) * 0.16;

    // cool edge-lit tint, thicker glass is milkier
    col = mix(col, vec3(0.78, 0.88, 0.96), 0.05 + 0.10 * thick * (1.0 - wipe));
    float vig = 1.0 - 0.28 * pow(length((uv - 0.5) * vec2(1.0, 0.9)) * 1.4, 2.0);
    fragColor = TDOutputSwizzle(vec4(col * vig * 0.95, 1.0));
}
