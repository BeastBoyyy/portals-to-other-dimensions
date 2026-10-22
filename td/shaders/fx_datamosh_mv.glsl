// datamosh, pass 1: block motion estimation. One texel per 16x16 pixel block of the camera. For every block it finds where that patch was in the previous
// camera frame (coarse search +-32 px in 8 px steps, then refined 4 / 2 / 1 px), the way a video encoder picks motion vectors.
// Output RG = offset in pixels from the block to its previous position (previous = current + offset). Flat or noisy blocks are forced to 0 (no motion).
// Inputs: 0 camera now, 1 camera one frame ago
layout(location = 0) out vec4 fragColor;

float curL[25];
vec2 cres;
vec2 centre;

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float sad(vec2 o) {
    float s = 0.0;
    for (int i = 0; i < 25; i++) {
        vec2 p = centre + (vec2(float(i % 5), float(i / 5)) - 2.0) * 3.2;
        s += abs(lum(textureLod(sTD2DInputs[1], (p + o) / cres, 1.0).rgb) - curL[i]);
    }
    return s / 25.0 + 0.00035 * length(o);
}

void main() {
    cres = vec2(textureSize(sTD2DInputs[0], 0));
    centre = (floor(gl_FragCoord.xy) + 0.5) * 16.0;
    for (int i = 0; i < 25; i++) {
        vec2 p = centre + (vec2(float(i % 5), float(i / 5)) - 2.0) * 3.2;
        curL[i] = lum(textureLod(sTD2DInputs[0], p / cres, 1.0).rgb);
    }
    float s0 = sad(vec2(0.0));
    vec2 best = vec2(0.0);
    float bs = s0;
    for (int oy = -4; oy <= 4; oy++)
        for (int ox = -4; ox <= 4; ox++) {
            if (ox == 0 && oy == 0) continue;
            vec2 o = vec2(float(ox), float(oy)) * 8.0;
            float s = sad(o);
            if (s < bs) { bs = s; best = o; }
        }
    for (int st = 0; st < 3; st++) {
        float step_ = 4.0 / float(1 << st);
        vec2 c = best;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                vec2 o = c + vec2(float(dx), float(dy)) * step_;
                float s = sad(o);
                if (s < bs) { bs = s; best = o; }
            }
    }
    if (bs > s0 - 0.006) best = vec2(0.0);                                      // not clearly better than "no motion"
    fragColor = vec4(best, 0.0, 1.0);
}
