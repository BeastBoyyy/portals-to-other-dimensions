// fx 'ascii': the room re-drawn as glowing text. Glyphs are picked by brightness from a density ramp (atlas = input 2),
// the person is full brightness and the background dim, and it reacts: glyphs flicker/scramble with hand speed, "code rain"
// falls through the columns, and a bright scan band sweeps down.
// Inputs: 0 camera, 1 person mask, 2 glyph atlas (R channel, n glyphs in one row).
// Vectors: uR = (time, speed, openness, twist 0..1)   uA = (cell_w_px, cell_h_px, n_glyphs, 0)
uniform vec4 uR;
uniform vec4 uA;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    vec2 cell = uA.xy;
    float n = uA.z;
    float time = uR.x, speed = uR.y;

    vec2 cid = floor(px / cell);
    vec2 f = fract(px / cell);
    vec2 cuv = (cid + 0.5) * cell / res;
    vec2 q = 0.28 * cell / res;
    vec3 c = 0.25 * (texture(sTD2DInputs[0], cuv + vec2(-q.x, -q.y)).rgb + texture(sTD2DInputs[0], cuv + vec2(q.x, -q.y)).rgb +
                     texture(sTD2DInputs[0], cuv + vec2(-q.x, q.y)).rgb + texture(sTD2DInputs[0], cuv + vec2(q.x, q.y)).rgb);
    float m = smoothstep(0.25, 0.75, texture(sTD2DInputs[1], cuv).r);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));

    float L = clamp((lum - 0.06) * 1.6, 0.0, 1.0);
    L = mix(L * 0.40, clamp(L * 1.15 + 0.08, 0.0, 1.0), m);               // person: full range; room: dimmer
    float flick = h21(cid + floor(time * 12.0));
    L = clamp(L + (flick - 0.5) * (0.10 + 0.55 * speed), 0.0, 1.0);        // fast hands scramble the glyphs

    float g = floor(L * (n - 1.0) + 0.5);
    float glyph = texture(sTD2DInputs[2], vec2((g + f.x) / n, f.y)).r;

    // code rain: some columns carry a bright falling head with a fading tail
    float colr = h21(vec2(cid.x, 7.0));
    float rows = res.y / cell.y;
    float headY = fract(time * (0.12 + colr * 0.35 + 0.4 * speed) + colr * 13.0) * (rows + 8.0) - 4.0;
    float dist = (rows - cid.y) - headY;                                     // top of the screen = 0
    float rain = step(0.78, colr) * (exp(-max(dist, 0.0) * 0.22) * step(0.0, dist)) * (0.5 + 0.8 * speed);
    float head = step(0.78, colr) * (1.0 - smoothstep(0.0, 1.2, abs(dist)));

    float sweep = pow(max(0.0, 1.0 - abs(fract(uv.y * 0.8 + time * 0.12) - 0.5) * 14.0), 2.0) * 0.35;   // scan band

    vec3 tint = mix(vec3(0.70, 1.0, 0.80), c * 1.9 + 0.12, 0.45);
    vec3 col = tint * glyph * (0.28 + 0.95 * L + sweep);
    col += vec3(0.4, 1.0, 0.6) * glyph * rain * 0.9 + vec3(1.0) * glyph * head * 0.9;
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
