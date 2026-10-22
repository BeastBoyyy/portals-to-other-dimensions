// GLSL TOP 'scene_moon' -- a glowing moon hanging at the top of the box, drawn BEHIND the person (the mask hides part of it:
// that is the white "ring" seen around the head in the reference). Black elsewhere.
// Inputs: 0 = person mask (display orientation).   Vectors: uM = (centre.x, centre.y, radius_uv_height, time)   uB = (glow, occlusion, 0, 0)
uniform vec4 uM;
uniform vec4 uB;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 p = (uv - uM.xy) * vec2(res.x / res.y, 1.0);
    float R = max(uM.z, 1e-4);
    float d = length(p) / R;
    float disc = 1.0 - smoothstep(0.96, 1.0, d);
    // surface: soft craters/maria from layered value noise, limb darkening, slow drift
    vec2 q = p / R * 2.2 + vec2(uM.w * 0.03, 0.0);
    float tex = 0.55 * vnoise(q * 2.0) + 0.3 * vnoise(q * 5.0) + 0.15 * vnoise(q * 11.0);
    float limb = sqrt(max(1.0 - d * d, 0.0));
    float lit = clamp(0.62 + 0.55 * limb - 0.35 * tex, 0.0, 1.0);
    vec3 moon = vec3(lit) * vec3(1.0, 0.99, 0.96) * disc;
    float glow = exp(-max(d - 1.0, 0.0) * 3.2) * (1.0 - disc) * uB.x;                 // halo
    vec3 col = moon + vec3(0.9, 0.93, 1.0) * glow * 0.4;
    float m = texture(sTD2DInputs[0], uv).r;                                           // person in front of the moon
    col *= 1.0 - clamp(smoothstep(0.35, 0.8, m) * uB.y, 0.0, 1.0);
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
