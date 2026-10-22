// fx 'datamosh': video-codec "P-frame" corruption. The picture on screen is NOT the camera: it is the previous output pushed along by the camera's motion vectors
// (16x16 blocks, crisp integer copies, so edges tear into macroblocks), and only a little fresh camera is mixed back in. Whatever moves therefore drags the old
// pixels along with it and leaves smeared, stretched trails; still areas slowly heal back to the live picture. Now and then a few blocks use a wrong vector,
// and a key-frame recovery band sweeps up the picture and restores it.
// Reactive: hand speed exaggerates the motion vectors, stops healing (the smear holds), and corrupts more blocks.
// Runs once per CAMERA frame (like a video codec); the renderer keeps the output buffer between frames.
// Inputs: 0 previous output, 1 camera now, 2 block motion vectors (nearest filtered)   uR = (time, speed, openness, twist)   uM = (refresh, sweep y, hold, 0)
uniform vec4 uR;
uniform vec4 uM;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 h22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }

void main() {
    vec2 res = uTDOutputInfo.res.zw;
    vec2 uv = vUV.st;
    vec2 blk = floor(gl_FragCoord.xy / 16.0);
    float speed = uR.y;
    vec2 o = texture(sTD2DInputs[2], uv).xy;
    float mag = length(o);
    o *= 1.0 + 0.9 * speed;                                                      // the codec "over-commits" to motion when you move fast

    float tq = floor(uR.x * 6.0);
    float hb = h21(blk + tq * 0.37);
    if (hb > 0.996 - 0.01 * speed) o += (h22(blk + tq) - 0.5) * 70.0;           // a few blocks use a wrong vector

    ivec2 ires = ivec2(res);
    vec3 prev = texelFetch(sTD2DInputs[0], clamp(ivec2(floor(gl_FragCoord.xy + o)), ivec2(0), ires - 1), 0).rgb;   // crisp integer copy: no resampling blur piles up

    vec3 cur = texture(sTD2DInputs[1], uv).rgb;
    float still = 1.0 - smoothstep(0.6, 2.5, mag);                              // blocks that do not move heal back to the live picture
    float refresh = uM.x * (1.0 - 0.6 * uM.z) + still * 0.07 * (1.0 - 0.6 * uM.z);
    float band = smoothstep(uM.y - 0.12, uM.y - 0.10, uv.y) * (1.0 - smoothstep(uM.y - 0.01, uM.y + 0.01, uv.y));   // key-frame recovery band
    refresh = max(refresh, band);

    vec3 col = mix(prev, cur, clamp(refresh, 0.0, 1.0));
    // codec colour quantisation: coarse chroma in 8x8 blocks, only where the picture is stale
    vec3 q = floor(col * 28.0 + 0.5) / 28.0;
    col = mix(col, q, 0.3 * (1.0 - refresh));
    fragColor = vec4(col, 1.0);
}
