// GLSL TOP 'scene_b' -- "asteroid": bright cratered body lit hard from one side on black. Procedural, no texture.
// (Optional: wire a public-domain NASA surface image into input 0 and set uB.x = 1 to use it as albedo.)
// Vectors: uA = (time, radius, rotation_speed, 0)   uB = (use_texture, contrast, centre_offset.xy)
uniform vec4 uA;
uniform vec4 uB;

layout(location = 0) out vec4 fragColor;

vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}

// Worley (cellular) distance: craters are bowls inside each cell, with a raised rim
float craters(vec3 p, float scale, out float rim) {
    p *= scale;
    vec3 ip = floor(p), fp = fract(p);
    float h = 0.0; rim = 0.0;
    for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec3 g = vec3(x, y, z);
        vec3 o = hash3(ip + g);
        vec3 d = g + o * 0.8 + 0.1 - fp;
        float rr = length(d);
        float R = 0.25 + 0.35 * o.x;
        h -= smoothstep(R, R * 0.2, rr) * (0.35 + 0.65 * o.y);
        rim += smoothstep(R * 0.2, R, rr) * smoothstep(R * 1.35, R, rr) * 0.4;
    }
    return h;
}

float height(vec3 p) {
    float rim1, rim2, rim3;
    float h = craters(p, 2.2, rim1) * 0.55 + craters(p + 3.7, 5.0, rim2) * 0.30 + craters(p + 9.1, 11.0, rim3) * 0.15;
    return h + (rim1 * 0.5 + rim2 * 0.3 + rim3 * 0.2) * 0.35;
}

void main() {
    vec2 uv = vUV.st;
    vec2 res = uTDOutputInfo.res.zw;
    vec2 p = (uv - 0.5 - uB.zw) * vec2(res.x / res.y, 1.0);              // uB.zw = centre offset (default 0)
    float R = uA.y;
    float r2 = dot(p, p);
    vec3 col = vec3(0.0);
    if (r2 < R * R) {
        float z = sqrt(R * R - r2);
        vec3 n = normalize(vec3(p, z));
        float a = uA.x * uA.z;                       // slow rotation about the vertical axis
        mat3 rot = mat3(cos(a), 0, -sin(a), 0, 1, 0, sin(a), 0, cos(a));
        vec3 sp = rot * n;
        float e = 0.012;
        float h0 = height(sp);
        vec3 grad = vec3(height(sp + vec3(e, 0, 0)) - h0, height(sp + vec3(0, e, 0)) - h0, height(sp + vec3(0, 0, e)) - h0) / e;
        vec3 sn = normalize(n - 0.55 * (transpose(rot) * (grad - sp * dot(grad, sp))));
        vec3 L = normalize(vec3(1.0, 0.25, 0.55));   // hard light from one side
        float diff = clamp(dot(sn, L), 0.0, 1.0);
        diff = pow(diff, 0.8);
        float albedo = 0.62 + 0.38 * smoothstep(-0.5, 0.1, h0);
        if (uB.x > 0.5) albedo = dot(texture(sTD2DInputs[0], vec2(atan(sp.z, sp.x) / 6.2831853 + 0.5, asin(clamp(sp.y, -1.0, 1.0)) / 3.14159 + 0.5)).rgb, vec3(0.333));
        float lit = diff * albedo;
        lit = pow(lit, 1.0 / max(uB.y, 0.2));
        float limb = smoothstep(0.0, 0.06, (R - sqrt(r2)) / R);   // antialiased silhouette
        col = vec3(lit) * vec3(1.0, 0.97, 0.93) * limb;
    }
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
