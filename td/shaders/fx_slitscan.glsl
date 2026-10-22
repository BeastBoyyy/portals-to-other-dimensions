// fx 'slitscan': time-displacement slit scan. Every pixel shows the room from a DIFFERENT moment: the live line sits on an axis through the box,
// and the further a pixel is from it the older the frame it shows (up to ~1.5 s back, blended between frames so time flows continuously).
// Anything that moves gets stretched into flowing ribbons; still parts of the room stay put. Far from the axis the picture is also pulled along the axis
// towards the live line (the classic photo-finish streak), colour channels sample slightly different moments (temporal chromatic fringes).
// Reactive: hand speed lengthens the time reach and the streaking; twisting the hand rotates the scan axis.
// Inputs: sHist = texture array of the last N camera frames (layer uH.x is the newest)   uR = (time, speed, openness, twist 0..1)   uC = (box centre uv, 0, 0)
uniform vec4 uR;
uniform vec4 uC;
uniform vec4 uH;                          // x = layer of the newest frame, y = layer count
uniform sampler2DArray sHist;

layout(location = 0) out vec4 fragColor;

vec3 frame(vec2 uv, float k) {            // the room k frames ago (k fractional: linear blend between neighbouring frames)
    float N = uH.y;
    k = clamp(k, 0.0, N - 2.0);
    float k0 = floor(k), f = k - k0;
    float l0 = mod(uH.x - k0 + N, N), l1 = mod(uH.x - k0 - 1.0 + N, N);
    return mix(texture(sHist, vec3(uv, l0)).rgb, texture(sHist, vec3(uv, l1)).rgb, f);
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 asp = vec2(res.x / res.y, 1.0);
    float time = uR.x, speed = uR.y, twist = uR.w;

    float th = (twist - 0.4) * 2.4 + 0.08 * sin(time * 0.21);                    // direction of the time gradient (0 = left/right: a vertical live line at rest)
    vec2 dir = vec2(cos(th), sin(th));
    vec2 p = (uv - uC.xy) * asp;
    float s = dot(p, dir);                                                       // signed distance across the live line
    float t = dot(p, vec2(-dir.y, dir.x));                                       // position along it
    float a = abs(s);

    float reach = mix(0.5, 0.75, speed);
    float ramp = pow(clamp(a / reach, 0.0, 1.0), 0.9);
    float kmax = 14.0 + 30.0 * (0.25 + 0.75 * speed);                            // frames of delay at the far edge
    float wob = (sin(s * 10.0 + time * 1.4) * 2.0 + sin(t * 6.5 - time * 0.8) * 1.5) * ramp;
    float k = ramp * kmax + wob;

    float st = smoothstep(0.10, 0.65, a) * (0.45 + 0.4 * speed);               // pull the sample position towards the live line
    vec2 suv = clamp(uv - dir * (s * st) / asp, 0.001, 0.999);

    float ca = 0.6 + 1.6 * speed;                                                // frames of colour-channel time offset
    vec3 col = vec3(frame(suv, k - ca).r, frame(suv, k).g, frame(suv, k + ca).b);

    float live = exp(-pow(s / 0.0035, 2.0));                                     // the live line itself
    col += vec3(0.35, 0.85, 1.0) * live * (0.35 + 0.4 * speed);
    col *= 1.0 - 0.18 * ramp;                                                    // time recedes into the dark
    col = (col - 0.5) * 1.07 + 0.5;
    fragColor = TDOutputSwizzle(vec4(max(col, 0.0), 1.0));
}
