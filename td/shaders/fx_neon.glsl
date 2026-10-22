// fx 'neon': glowing multi-scale edge lines on black with a drifting hue, a faint thermal wash on the person, and a hot white
// silhouette. Reactive: the hue cycle speeds up and the lines thicken with hand speed.
// Inputs: 0 camera, 1 person mask.   Vectors: uR = (time, speed, openness, twist 0..1)
uniform vec4 uR;

layout(location = 0) out vec4 fragColor;

vec3 hsv(float h, float s, float v) {
    vec3 k = clamp(abs(fract(vec3(h) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}
float lumAt(vec2 uv) { return dot(texture(sTD2DInputs[0], uv).rgb, vec3(0.299, 0.587, 0.114)); }
float sobel(vec2 uv, vec2 e) {
    float tl = lumAt(uv + vec2(-e.x, e.y)), t = lumAt(uv + vec2(0, e.y)), tr = lumAt(uv + vec2(e.x, e.y));
    float l = lumAt(uv + vec2(-e.x, 0)), r = lumAt(uv + vec2(e.x, 0));
    float bl = lumAt(uv + vec2(-e.x, -e.y)), b = lumAt(uv + vec2(0, -e.y)), br = lumAt(uv + vec2(e.x, -e.y));
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br, gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
    return length(vec2(gx, gy));
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    float time = uR.x, speed = uR.y;
    float k = 1.0 + 1.5 * speed;
    float e1 = sobel(uv, 1.0 * k / res) * 0.9;
    float e2 = sobel(uv, 3.0 * k / res) * 0.6;
    float e3 = sobel(uv, 7.0 * k / res) * 0.45;
    float m = smoothstep(0.2, 0.8, texture(sTD2DInputs[1], uv).r);
    float lum = lumAt(uv);

    float hue = fract(time * (0.05 + 0.25 * speed) + lum * 0.5 + uv.x * 0.25);
    vec3 line = hsv(hue, 0.85, 1.0) * (e1 + e2) + hsv(hue + 0.12, 0.7, 1.0) * e3;
    vec3 thermal = hsv(0.62 - lum * 0.62, 0.9, 0.5) * lum * m * 0.30;            // cold blue -> hot red wash on the person
    vec2 e = 2.0 / res;
    float rim = abs(texture(sTD2DInputs[1], uv + vec2(e.x, 0)).r - texture(sTD2DInputs[1], uv - vec2(e.x, 0)).r) +
                abs(texture(sTD2DInputs[1], uv + vec2(0, e.y)).r - texture(sTD2DInputs[1], uv - vec2(0, e.y)).r);
    vec3 col = line * (0.5 + 0.9 * m) + thermal + vec3(1.0) * rim * 1.6;
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
