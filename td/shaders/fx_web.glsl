// fx 'web': the person as a luminous line-web: horizontal scan lines are pushed up by brightness (a "mountain" portrait), crossed by
// faint vertical threads that sag toward the figure. Reactive: waves travel along the lines with hand speed, amplitude breathes.
// Inputs: 0 camera, 1 person mask.   Vectors: uR = (time, speed, openness, twist 0..1)   uA = (row_px, amp_px, col_px, 0)
uniform vec4 uR;
uniform vec4 uA;

layout(location = 0) out vec4 fragColor;

float height(vec2 uv) {
    vec3 c = texture(sTD2DInputs[0], uv).rgb;
    float m = smoothstep(0.15, 0.8, texture(sTD2DInputs[1], uv).r);
    return dot(c, vec3(0.299, 0.587, 0.114)) * (0.10 + 0.90 * m);
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = uv * res;
    float time = uR.x, speed = uR.y;
    float rs = uA.x, amp = uA.y * (0.85 + 0.5 * speed), cs = uA.z;

    float acc = 0.0;
    for (int k = -4; k <= 2; k++) {
        float rowY = (floor(px.y / rs) + float(k)) * rs;
        float h = height(vec2(px.x, rowY) / res);
        float wave = sin(px.x * 0.018 + time * (2.0 + 6.0 * speed) + rowY * 0.04) * (0.04 + 0.30 * speed);
        float yLine = rowY + (h + wave * h) * amp;
        float d = abs(px.y - yLine);
        float line = smoothstep(1.35, 0.25, d) * (0.22 + 0.78 * h);
        acc = max(acc, line);
    }
    float web = 0.0;
    for (int k = -1; k <= 1; k++) {                                  // vertical threads, bent horizontally by the local brightness gradient
        float colX = (floor(px.x / cs) + float(k)) * cs;
        float hc = height(vec2(colX, px.y) / res);
        float hr = height(vec2(colX + 6.0, px.y) / res), hl = height(vec2(colX - 6.0, px.y) / res);
        float xLine = colX + (hr - hl) * cs * 2.2 + sin(px.y * 0.03 + time * 2.0) * 2.0 * speed;
        web = max(web, smoothstep(1.0, 0.2, abs(px.x - xLine)) * (0.10 + 0.30 * hc));
    }
    vec3 col = vec3(0.86, 0.95, 1.0) * acc + vec3(0.55, 0.75, 1.0) * web;
    col += vec3(0.4, 0.65, 1.0) * acc * acc * 0.35;                    // hot core on the brightest lines
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
