// fx 'moire': the room seen through layered fine line screens whose interference makes big, slow, flowing moire fringes.
// Physically a moire is the low-pass of the product of two gratings, i.e. a cosine of the PHASE DIFFERENCE, so the fringes are computed analytically from that
// difference: perfectly smooth, no aliasing at any window scale, and continuous in time. Three screens (A, B and a concentric ring screen centred on the box)
// beat against each other; the camera image bends the phase of B, so shapes in the room show up as flowing contour fringes. A faint copy of the real fine
// line texture is mixed back in (anti-aliased from derivatives) so it still looks like a physical screen, and each colour channel has its own tiny angle
// offset -> soft chromatic fringes.
// Reactive: hand speed opens the angle between the screens (fringes tighten and race); twisting rotates the screens and changes their pitch.
// Inputs: 0 camera (mip-mapped)   uR = (time, speed, openness, twist 0..1)   uC = (box centre uv, 0, 0)
uniform vec4 uR;
uniform vec4 uC;

layout(location = 0) out vec4 fragColor;

const float TAU = 6.2831853;
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float screen(float x) {                                   // fine line screen, anti-aliased: 1 in the open band, 0 on the line
    float v = abs(fract(x) - 0.5) * 2.0;
    float w = max(fwidth(v), 1e-4) * 1.2;
    return 1.0 - smoothstep(0.55 - w, 0.55 + w, v);
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    float time = uR.x, speed = uR.y, twist = uR.w;

    vec3 camc = textureLod(sTD2DInputs[0], uv, 0.5).rgb;
    float L = lum(textureLod(sTD2DInputs[0], uv, 2.0).rgb);

    float per = 4.4 + 2.4 * twist;                                              // line pitch (px)
    float a0 = 0.50 + 0.8 * twist + 0.05 * sin(time * 0.17);
    float dth = 0.020 + 0.012 * sin(time * 0.31) + 0.045 * smoothstep(0.0, 0.9, speed);   // small angle between the screens = big fringes
    float ph = (L - 0.5) * (1.8 + 2.0 * speed);                                // the image bends screen B
    vec2 q = (uv - uC.xy) * vec2(res.x / res.y, 1.0);
    float rr = length(q) * res.y;
    float ringMix = smoothstep(0.65, 0.05, length(q));                          // rings matter near the box centre

    vec3 T;
    float fine = 0.0;
    for (int c = 0; c < 3; c++) {
        float da = (float(c) - 1.0) * 0.0035;                                   // per-channel angle offset -> chromatic fringes
        vec2 dA = vec2(cos(a0 + da), sin(a0 + da));
        vec2 dB = vec2(cos(a0 + dth + da * 1.6), sin(a0 + dth + da * 1.6));
        float pA = dot(px, dA) / per;
        float pB = dot(px, dB) / per + ph;
        float pR = rr / (per * 1.035) + time * 0.10 - ph * 0.6;
        float f1 = 0.5 + 0.5 * cos(TAU * (pA - pB));                             // A x B beat
        float f2 = 0.5 + 0.5 * cos(TAU * (pA - pR));                             // A x rings beat
        float f3 = 0.5 + 0.5 * cos(TAU * (pB - pR * 0.5 - 0.37));                // third family
        T[c] = mix(f1, 0.5 * (f2 + f3), ringMix * 0.8);
        if (c == 1) fine = screen(pA) * screen(pB);
    }
    T = pow(T, vec3(1.6));                                                       // sharpen the fringes a little
    T = mix(T, vec3(fine), 0.22);                                                // physical line texture, faint

    vec3 bg = vec3(0.008, 0.013, 0.030);
    vec3 lit = camc * 1.6 + vec3(0.07, 0.10, 0.16);
    vec3 col = bg + lit * T * 1.25;
    col += lit * dot(T, vec3(0.3333)) * 0.10;                                    // glow of the bright fringes
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
