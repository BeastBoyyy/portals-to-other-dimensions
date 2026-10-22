// fx 'bitmap': crisp 1-bit ordered-dither bitmap (Bayer 8x8) of the scene. The person is rendered with full tonal range, the room
// recedes. Reactive: the dither threshold travels with hand speed, scanline bands slide sideways, pixels grow as the hand twists.
// Inputs: 0 camera, 1 person mask.   Vectors: uR = (time, speed, openness, twist 0..1)   uA = (pixel_px, contrast, 0, 0)
uniform vec4 uR;
uniform vec4 uA;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

float bayer8(ivec2 p) {                                   // 8x8 ordered-dither matrix, 0..1
    int x = p.x & 7, y = p.y & 7;
    int a = x ^ y;
    int v = ((a & 1) << 5) | ((y & 1) << 4) | ((a & 2) << 2) | ((y & 2) << 1) | ((a & 4) >> 1) | ((y & 4) >> 2);
    return (float(v) + 0.5) / 64.0;
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    float time = uR.x, speed = uR.y, twist = uR.w;
    float ps = max(1.0, floor(uA.x * (1.0 + 0.8 * twist) + 0.5));        // chunkier pixels as the hand twists
    vec2 px = uv * res;
    float band = floor(px.y / 18.0);
    float tq = floor(time * 10.0);
    float slide = (h21(vec2(band, tq)) - 0.5) * 30.0 * speed * step(0.65, h21(vec2(band + 5.0, tq)));   // glitchy sliding bands (px)
    vec2 gp = floor((px + vec2(slide, 0.0)) / ps);
    vec2 suv = (gp + 0.5) * ps / res;
    vec3 c = texture(sTD2DInputs[0], suv).rgb;
    float m = smoothstep(0.2, 0.8, texture(sTD2DInputs[1], suv).r);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float L = clamp((lum - 0.5) * uA.y + 0.5, 0.0, 1.0);
    L = mix(L * 0.30, clamp(L * 1.1 + 0.06, 0.0, 1.0), m);

    float thr = bayer8(ivec2(gp) + ivec2(int(floor(time * 18.0 * speed)), 0));   // the pattern crawls with hand speed
    float on = step(thr, L);

    // 1-bit contour of the silhouette (hard white line)
    vec2 e = ps / res;
    float edge = abs(texture(sTD2DInputs[1], suv + vec2(e.x, 0)).r - texture(sTD2DInputs[1], suv - vec2(e.x, 0)).r) +
                 abs(texture(sTD2DInputs[1], suv + vec2(0, e.y)).r - texture(sTD2DInputs[1], suv - vec2(0, e.y)).r);
    on = max(on, step(0.35, edge));
    vec3 ink = mix(vec3(0.96, 0.98, 1.0), vec3(1.0, 0.92, 0.75), 0.5 * (1.0 - m));    // warm paper-white for the room
    fragColor = TDOutputSwizzle(vec4(ink * on, 1.0));
}
