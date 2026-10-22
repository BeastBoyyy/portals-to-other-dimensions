// fx 'crt': the room on a curved cathode-ray tube. Barrel curvature, a gaussian beam per scan line whose height grows with brightness, a phosphor triad mask,
// horizontal beam softness, colour convergence error that grows towards the corners, halation (light bleeding from the glass, from the camera mip chain),
// a slow rolling brightness bar, mains flicker, line-to-line sync wobble and fine grain.
// Reactive: hand speed raises sync tearing (rows slide sideways), convergence error and the roll bar.
// Inputs: 0 camera (mip-mapped)   uR = (time, speed, openness, twist 0..1)
uniform vec4 uR;

layout(location = 0) out vec4 fragColor;

float h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    float time = uR.x, speed = uR.y;

    // barrel distortion
    vec2 cc = uv * 2.0 - 1.0;
    float r2 = dot(cc * vec2(0.92, 1.0), cc * vec2(0.92, 1.0));
    vec2 cu = cc * (1.0 + 0.085 * r2 + 0.02 * r2 * r2) * 0.97 * 0.5 + 0.5;
    vec2 edge = smoothstep(vec2(0.0), vec2(0.004), cu) * (1.0 - smoothstep(vec2(0.996), vec2(1.0), cu));
    float inside = edge.x * edge.y;

    // scan line coordinates (CRT resolution: 120 lines = 6 px per line at 720p, so they survive the portal's perspective resampling)
    const float LINES = 120.0;
    float yy = cu.y * LINES;
    float line = floor(yy);
    float tq = floor(time * 30.0);
    // sync wobble + tearing
    float wob = (h21(vec2(line, tq)) - 0.5) * 0.0007 + 0.0012 * sin(line * 0.11 + time * 2.0);
    float band = floor(cu.y * 14.0);
    float tear = step(0.985 - 0.1 * speed, h21(vec2(band, tq))) * (h21(vec2(band + 3.0, tq)) - 0.5) * (0.02 + 0.06 * speed);
    float xs = cu.x + wob + tear;

    // convergence error (R and B guns land slightly off, more at the corners)
    float conv = (0.0007 + 0.0016 * r2 + 0.0014 * speed);
    vec3 col;
    float dx = 1.15 / res.x;
    // horizontal beam softness: 5 taps
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for (int i = -2; i <= 2; i++) {
        float w = exp(-float(i * i) * 0.45);
        vec2 q = vec2(xs + float(i) * dx, (line + 0.5) / LINES);
        acc += w * vec3(texture(sTD2DInputs[0], q + vec2(-conv, 0.0)).r, texture(sTD2DInputs[0], q).g, texture(sTD2DInputs[0], q + vec2(conv, 0.0)).b);
        wsum += w;
    }
    col = acc / wsum;
    col = pow(col, vec3(1.45)) * 1.05;                                           // CRT gamma response: punchy blacks

    // beam profile across the scan line: bright lines are fat, dark lines thin
    float d = fract(yy) - 0.5;
    float L = lum(col);
    float sigma = mix(0.15, 0.36, clamp(L * 1.3, 0.0, 1.0));
    float beam = exp(-(d * d) / (2.0 * sigma * sigma));
    col *= mix(0.04, 1.0, beam) * 1.35;

    // phosphor triad (6 px period: 2 px per colour)
    float xm = fract(gl_FragCoord.x / 6.0);
    vec3 mask = vec3(smoothstep(0.0, 0.2, xm) * (1.0 - smoothstep(0.30, 0.45, xm)),
                     smoothstep(0.30, 0.45, xm) * (1.0 - smoothstep(0.63, 0.78, xm)),
                     smoothstep(0.63, 0.78, xm) * (1.0 - smoothstep(0.95, 1.0, xm)));
    mask = mix(vec3(1.0), mask * 2.3 + 0.22, 0.75);
    col *= mask;

    // halation: light scattered inside the glass
    vec3 glow = textureLod(sTD2DInputs[0], cu, 4.0).rgb + 0.6 * textureLod(sTD2DInputs[0], cu, 5.5).rgb;
    col += glow * 0.34 * vec3(0.85, 1.0, 1.1);

    // rolling bar, flicker, grain, vignette
    float bar = smoothstep(0.0, 0.25, fract(cu.y * 0.9 - time * (0.07 + 0.2 * speed))) ;
    col *= 0.93 + 0.07 * bar;
    col *= 0.975 + 0.025 * sin(time * 100.0) + 0.012 * sin(time * 7.3);
    col += (h21(gl_FragCoord.xy + tq) - 0.5) * 0.035;
    float vig = 1.0 - 0.45 * pow(length(cc * vec2(0.9, 1.0)) * 0.78, 2.4);
    col *= vig;
    col *= mix(vec3(1.0), vec3(0.93, 1.02, 1.0), 0.6);                           // slightly green-cool phosphor white
    col = col / (1.0 + 0.55 * col);                                               // soft shoulder: bright windows do not clip to flat white
    col *= 1.45;
    fragColor = TDOutputSwizzle(vec4(max(col, 0.0) * inside, 1.0));
}
