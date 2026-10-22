// GLSL TOP 'scene_a' -- "particle self-view"  (needs user verification in TD)
// Inputs: 0 = camera (display orientation), 1 = person mask (display orientation, 0..1)
// Vectors: uA = (time, cell_px, depth_px, glitch)    uB = (edge_noise, rim_gain, dot_px, 0)
// Point budget = (width/cell)*(height/cell): cell 4 px at 1280x720 = 57,600 points. Raise cell_px to scale down.
uniform vec4 uA;
uniform vec4 uB;
uniform vec4 uD;            // (motion.x, motion.y, speed 0..1, twist 0..1): reactive; zeros = calm

layout(location = 0) out vec4 fragColor;

float h11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    float time = uA.x, cell = max(uA.y, 1.5), depth = uA.z, glitch = uA.w;
    float edgeNoise = uB.x, rimGain = uB.y, dotPx = uB.z;

    // low-amplitude time-driven horizontal glitch bands (as in the reference)
    float band = floor(uv.y * 36.0);
    float tq = floor(time * 5.0);
    float gOn = step(0.965, h21(vec2(band, tq)));
    float shift = (h21(vec2(band + 13.0, tq)) - 0.5) * 0.02 * glitch * gOn;

    vec2 cellId = floor(px / cell);
    vec2 jitter = vec2(h21(cellId), h21(cellId + 7.7)) - 0.5;
    vec2 centre = (cellId + 0.5 + jitter * 0.55) * cell;
    // reactive: each particle samples the scene from where the hand WAS (random lag per particle) -> comet streaks trailing the motion
    vec2 lagpx = uD.xy * uD.z * (6.0 + 150.0 * h21(cellId + 5.0));
    vec2 suv = clamp((centre - lagpx) / res + vec2(shift, 0.0), 0.0, 1.0);

    vec3 cam = texture(sTD2DInputs[0], suv).rgb;
    float m = smoothstep(0.15, 0.85, texture(sTD2DInputs[1], suv).r);   // person mask, sharpened
    float lum = dot(cam, vec3(0.299, 0.587, 0.114));

    // slight z displacement from luminance and mask: the point cloud has volume
    float edge0 = 4.0 * m * (1.0 - m);
    float streak = (h21(cellId + 21.0) - 0.35) * edge0 * cell * 5.0 * edgeNoise;   // grains drift sideways off the silhouette
    vec2 dpos = centre + vec2(streak, (lum - 0.5) * depth + (m - 0.5) * depth * 0.5);

    // dissolve into grain at the silhouette: keep probability follows the mask, noisy where the mask is soft
    float edge = 4.0 * m * (1.0 - m);
    float r = h21(cellId * 1.37 + floor(time * 8.0));
    float keep = step(r, clamp(m + (h21(cellId + 3.1) - 0.5) * edge * edgeNoise * 2.0, 0.0, 1.0));
    keep *= step(0.04, m);

    // point sprite: soft round dot, 1-2 px
    float rad = dotPx * (0.6 + 0.8 * h21(cellId + 11.0)) * (1.0 + 0.9 * uD.z * h21(cellId + 17.0));
    float dd = length(px - dpos);
    float dotMask = (1.0 - smoothstep(rad - 0.5, rad + 0.5, dd)) * keep;

    // colour: camera colour pushed toward warm white / pink with contrast
    vec3 c = pow(clamp(cam, 0.0, 1.0), vec3(0.8)) * 1.45;            // natural colours (skin stays warm), brightened
    c = mix(c, vec3(1.0, 0.93, 0.95) * (0.5 + lum), 0.25);
    c = clamp(c + 0.12, 0.0, 1.0);

    // rim light: glowing white outline of the silhouette (gradient of the mask, 3 radii)
    vec2 e1 = 1.0 / res;
    float rim = 0.0;
    for (int k = 1; k <= 2; k++) {
        vec2 o = e1 * float(k);
        float gx = texture(sTD2DInputs[1], uv + vec2(o.x, 0)).r - texture(sTD2DInputs[1], uv - vec2(o.x, 0)).r;
        float gy = texture(sTD2DInputs[1], uv + vec2(0, o.y)).r - texture(sTD2DInputs[1], uv - vec2(0, o.y)).r;
        rim += length(vec2(gx, gy)) * 0.5 / float(k);
    }
    rim = clamp(rim * rimGain, 0.0, 1.0);

    vec3 col = c * dotMask + vec3(1.0) * rim * 0.55;
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
