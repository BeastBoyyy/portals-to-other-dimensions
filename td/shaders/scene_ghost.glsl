// GLSL TOP 'scene_ghost' -- the white "ghost" of the person: only the contour lines of the silhouette, repeated as vertical streak
// echoes (like the hollow white face at ~3 s in the reference). Black elsewhere.
// Inputs: 0 = camera, 1 = person mask.   Vectors: uA = (time, echo_len_uv, gain, 0)
uniform vec4 uA;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

float contour(vec2 uv, vec2 e) {
    float gx = texture(sTD2DInputs[1], uv + vec2(e.x, 0)).r - texture(sTD2DInputs[1], uv - vec2(e.x, 0)).r;
    float gy = texture(sTD2DInputs[1], uv + vec2(0, e.y)).r - texture(sTD2DInputs[1], uv - vec2(0, e.y)).r;
    return length(vec2(gx, gy));
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 e = 1.5 / res;
    float time = uA.x, echo = uA.y, gain = uA.z;
    float acc = 0.0;
    for (int k = 0; k < 8; k++) {                                       // streak: sample the contour above/below with decay
        float fk = float(k) / 7.0;
        float jit = (h21(vec2(floor(uv.x * res.x / 2.0), floor(time * 6.0) + float(k))) - 0.5) * 0.002;
        acc += contour(uv + vec2(jit, -fk * echo), e) * (1.0 - fk) * 0.8;
        acc += contour(uv + vec2(jit, fk * echo * 0.3), e) * (1.0 - fk) * 0.25;
    }
    float m = texture(sTD2DInputs[1], uv).r;
    float lum = dot(texture(sTD2DInputs[0], uv).rgb, vec3(0.299, 0.587, 0.114));
    float fill = smoothstep(0.4, 0.9, m) * (0.05 + 0.10 * lum);        // faint interior so the head reads as a hollow shell
    float v = clamp(acc * gain + fill, 0.0, 1.0);
    fragColor = TDOutputSwizzle(vec4(vec3(v), 1.0));
}
