// fx 'pixelsort': pixel sorting. The picture is cut into vertical spans (random length per 3 px column, scrolling slowly); wherever a span's brightness lies
// inside a moving threshold window it is REPLACED by its own pixels sorted dark -> bright, which melts the image into long molten streaks; the rest of the room stays intact.
// The sort is computed per pixel on the GPU: K samples of the span are put in order by an odd-even sorting network, and the pixel reads its place in that order
// (linearly interpolated, so it is smooth in space and in time). Samples come from a mip level of the camera so sensor noise does not make the streaks flicker.
// Reactive: hand speed widens the brightness window, speeds the scrolling, shifts a few spans sideways (glitch) and makes streaks longer.
// Inputs: 0 camera (mip-mapped)   uR = (time, speed, openness, twist 0..1)
uniform vec4 uR;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

const int K = 16;
const float CELL = 3.0;                                                          // rendered at HALF resolution: 3 px here = 6 px on screen
vec4 v[K];                                                                       // rgb + luma (every access below uses a constant index -> registers, no scratch memory)
#define CX(i, j) { vec4 a_ = v[i], b_ = v[j]; bool s_ = a_.w > b_.w; v[i] = s_ ? b_ : a_; v[j] = s_ ? a_ : b_; }
#define LD(i) { float y = start + (float(i) + 0.5) / float(K) * S; vec3 c = textureLod(sTD2DInputs[0], vec2(sx + shiftx, clamp(y / res.y, 0.0, 1.0)), 1.5).rgb; v[i] = vec4(c, lum(c)); avg += v[i].w / float(K); }
#define RD(i) sorted += v[i].rgb * max(0.0, 1.0 - abs(f - float(i)));

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    float time = uR.x, speed = uR.y, twist = uR.w;

    // spans follow the picture: a span is a vertical run of 6 px cells whose brightness lies in a moving window (max 20 cells each way)
    float gx = floor(px.x / 2.0);                                               // 2 px columns (= 4 px on screen)
    float sx = (gx * 2.0 + 1.0) / res.x;
    float lo = 0.12 + 0.22 * (0.5 + 0.5 * sin(time * 0.37)) - 0.10 * speed + 0.06 * sin(gx * 0.05 + time * 0.2);   // a brightness WINDOW that sweeps through the picture
    float hi = lo + 0.42 + 0.2 * speed;
    float cell = floor(px.y / CELL);
    float up = 0.0, dn = 0.0, goUp = 1.0, goDn = 1.0;
    for (int i = 1; i <= 20; i++) {
        float lu = lum(textureLod(sTD2DInputs[0], vec2(sx, (cell + float(i) + 0.5) * CELL / res.y), 2.5).rgb);
        float ld = lum(textureLod(sTD2DInputs[0], vec2(sx, (cell - float(i) + 0.5) * CELL / res.y), 2.5).rgb);
        goUp *= step(lo, lu) * (1.0 - step(hi, lu)); goDn *= step(lo, ld) * (1.0 - step(hi, ld));
        up += goUp; dn += goDn;
    }
    float start = (cell - dn) * CELL;
    float S = (up + dn + 1.0) * CELL;
    float tpos = clamp((px.y - start) / S, 0.0, 1.0);
    float spanId = h21(vec2(gx, floor(start / CELL)));
    float gl = h21(vec2(gx, floor(start / CELL) + 7.0));
    float shiftx = step(0.93 - 0.1 * speed, gl) * (h21(vec2(gx, floor(start / CELL) + 3.0)) - 0.5) * (0.01 + 0.04 * speed);   // some spans slide sideways
    float own = lum(textureLod(sTD2DInputs[0], vec2(sx, (cell + 0.5) * CELL / res.y), 2.5).rgb);
    float avg = 0.0;
    LD(0) LD(1) LD(2) LD(3) LD(4) LD(5) LD(6) LD(7) LD(8) LD(9) LD(10) LD(11) LD(12) LD(13) LD(14) LD(15)
    // Batcher odd-even merge sorting network (63 compare-exchanges), ascending luma
    CX(0,1) CX(2,3) CX(0,2) CX(1,3) CX(1,2) CX(4,5) CX(6,7) CX(4,6)
    CX(5,7) CX(5,6) CX(0,4) CX(2,6) CX(2,4) CX(1,5) CX(3,7) CX(3,5)
    CX(1,2) CX(3,4) CX(5,6) CX(8,9) CX(10,11) CX(8,10) CX(9,11) CX(9,10)
    CX(12,13) CX(14,15) CX(12,14) CX(13,15) CX(13,14) CX(8,12) CX(10,14) CX(10,12)
    CX(9,13) CX(11,15) CX(11,13) CX(9,10) CX(11,12) CX(13,14) CX(0,8) CX(4,12)
    CX(4,8) CX(2,10) CX(6,14) CX(6,10) CX(2,4) CX(6,8) CX(10,12) CX(1,9)
    CX(5,13) CX(5,9) CX(3,11) CX(7,15) CX(7,11) CX(3,5) CX(7,9) CX(11,13)
    CX(1,2) CX(3,4) CX(5,6) CX(7,8) CX(9,10) CX(11,12) CX(13,14)
    float f = tpos * float(K - 1);
    vec3 sorted = vec3(0.0);
    RD(0) RD(1) RD(2) RD(3) RD(4) RD(5) RD(6) RD(7) RD(8) RD(9) RD(10) RD(11) RD(12) RD(13) RD(14) RD(15)

    float act = smoothstep(lo - 0.02, lo + 0.02, own) * (1.0 - smoothstep(hi - 0.02, hi + 0.02, own)) * smoothstep(3.0, 6.5, (up + dn + 1.0)) * smoothstep(0.08, 0.2, spanId + 0.05);

    vec3 orig = texture(sTD2DInputs[0], uv).rgb;
    vec3 sorted_c = sorted * 1.08;
    float lm = lum(sorted_c);
    sorted_c = mix(vec3(lm), sorted_c, 1.25);                                    // a touch more colour in the streaks
    vec3 col = mix(orig * 0.82, sorted_c, act);
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
