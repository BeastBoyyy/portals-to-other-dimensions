// GLSL TOP 'composite'  (needs user verification in TD)
// The portal is a glass BOX seen in perspective, hung from the fingertips (as in the reference clip):
//   front face Q  = thumb tips (bottom) + the raised-finger anchors (top), 2 px white outline
//   back face B   = Q scaled toward the screen centre (`inset`): the dimension lives here (black + effect)
//   glass LID     = polygon A0 A1 L1 L0: between the anchors and the middle/ring fingertips, with the angled wings
//   glass mode    = clear box (room visible through every face)
// Inputs: 0 = camera (display orientation), 1 = scene (mixed effects), 2 = bloom of the scene
// Vectors: uQA=(q0,q1) uQB=(q2,q3) front quad in draw order | uLA=(A0,A1) uLB=(L0,L1) | uP1=(alpha, inset, time, tunnel_on)
//          uP2=(parallax.xy, bloom_gain, outline_px) | uP3=(glass 0..1, lid_on 0..1, 0, 0)       all in UV, bottom-left origin
uniform vec4 uQA;
uniform vec4 uQB;
uniform vec4 uLA;
uniform vec4 uLB;
uniform vec4 uP1;
uniform vec4 uP2;
uniform vec4 uP3;
uniform vec4 uIn;           // back-face scale per front-face corner (draw order): the 3D twist of the box
uniform vec4 uG;            // (speed 0..1, time, 0, 0): hand speed -> glitch bands + RGB split on the dimension

layout(location = 0) out vec4 fragColor;

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    return length(pa - ba * h);
}

// signed distance (pixels, >0 inside) to a convex quad given in either winding
float quadSDF(vec2 p, vec2 q0, vec2 q1, vec2 q2, vec2 q3) {
    vec2 q[4] = vec2[4](q0, q1, q2, q3);
    float area = 0.0;
    for (int i = 0; i < 4; i++) { vec2 a = q[i], b = q[(i + 1) % 4]; area += a.x * b.y - b.x * a.y; }
    float s = area >= 0.0 ? 1.0 : -1.0;
    float inside = 1e9, dmin = 1e9;
    for (int i = 0; i < 4; i++) {
        vec2 a = q[i], b = q[(i + 1) % 4];
        vec2 e = b - a;
        float side = s * (e.x * (p.y - a.y) - e.y * (p.x - a.x)) / max(length(e), 1e-9);
        inside = min(inside, side);
        dmin = min(dmin, segDist(p, a, b));
    }
    return inside > 0.0 ? dmin : -dmin;
}

float outline(float dist, float w) { return 1.0 - smoothstep(w * 0.5 - 0.5, w * 0.5 + 0.5, dist); }

void main() {
    vec2 uv = vUV.st;
    vec2 S = uTDOutputInfo.res.zw;                   // (width, height) in pixels: distances below are in pixels
    float alpha = uP1.x;
    float glass = clamp(uP3.x, 0.0, 1.0), lidOn = uP3.y;
    vec2 q0 = uQA.xy, q1 = uQA.zw, q2 = uQB.xy, q3 = uQB.zw;
    vec2 a0 = uLA.xy, a1 = uLA.zw, l0 = uLB.xy, l1 = uLB.zw;
    vec2 P = uv * S;

    float dQ = quadSDF(P, q0 * S, q1 * S, q2 * S, q3 * S);
    float mQ = smoothstep(-0.5, 0.5, dQ);
    float dL = quadSDF(P, a0 * S, a1 * S, l1 * S, l0 * S);
    float mL = smoothstep(-0.5, 0.5, dL) * lidOn;

    vec3 cam = texture(sTD2DInputs[0], uv).rgb;
    vec2 puv = uv + uP2.xy;                          // parallax: the dimension shifts a little with the box tilt / position
    // reactive glitch: horizontal bands shift and the colour channels split as the hands move fast
    float gband = floor(puv.y * 30.0), gtq = floor(uG.y * 14.0);
    float gh = fract(sin(dot(vec2(gband, gtq), vec2(12.9898, 78.233))) * 43758.5453);
    float gate = step(1.0 - 0.55 * uG.x, gh);
    vec2 guv = puv + vec2((fract(gh * 7.31) - 0.5) * 0.10 * uG.x * gate, 0.0);
    vec2 ca = vec2(0.005 * uG.x * (0.3 + gate), 0.0);
    vec3 content = vec3(texture(sTD2DInputs[1], guv + ca).r, texture(sTD2DInputs[1], guv).g, texture(sTD2DInputs[1], guv - ca).b)
                 + texture(sTD2DInputs[2], puv).rgb * uP2.z;
    vec3 clear = cam * 1.03 + vec3(0.015);           // glass: the room, very slightly lifted

    // back face: Q scaled toward the camera's principal point
    vec2 V = vec2(0.5, 0.5);
    vec2 b0 = V + (q0 - V) * uIn.x, b1 = V + (q1 - V) * uIn.y, b2 = V + (q2 - V) * uIn.z, b3 = V + (q3 - V) * uIn.w;
    float mB = smoothstep(-0.5, 0.5, quadSDF(P, b0 * S, b1 * S, b2 * S, b3 * S));
    vec3 wall = cam * 0.60 + content * 0.9 + vec3(0.03, 0.03, 0.04);         // side walls: transparent smoked glass, dimension shows through
    vec3 inner = mix(mix(wall, content, mB), clear, glass);
    vec3 lidc = mix(content + vec3(0.025), clear + vec3(0.05), glass);       // the lid is the box ceiling: black + the moon

    vec3 col = cam;
    col = mix(col, lidc, mL * alpha);
    col = mix(col, inner, mQ * alpha);
    col = mix(col, vec3(1.0), 0.05 * mQ * (1.0 - mB) * alpha);                // faint glass sheen on the walls

    // edges: back-face outline + 4 connecting edges (translucent), front outline 2 px, lid outline 1.4 px
    float lines = 0.0;
    if (uP1.w > 0.5) {
        float dl = min(min(segDist(P, b0 * S, b1 * S), segDist(P, b1 * S, b2 * S)),
                       min(segDist(P, b2 * S, b3 * S), segDist(P, b3 * S, b0 * S)));
        float dc = min(min(segDist(P, q0 * S, b0 * S), segDist(P, q1 * S, b1 * S)),
                       min(segDist(P, q2 * S, b2 * S), segDist(P, q3 * S, b3 * S)));
        lines = (outline(dl, 1.2) * 0.5 + outline(dc, 1.2) * 0.35) * mQ * alpha;
    }
    float dLid = min(min(segDist(P, a0 * S, l0 * S), segDist(P, l0 * S, l1 * S)), segDist(P, l1 * S, a1 * S));
    float lidLine = outline(dLid, 1.4) * lidOn * alpha;
    float frontLine = outline(abs(dQ), uP2.w) * alpha;
    col = mix(col, vec3(1.0), clamp(frontLine + lines + lidLine * 0.85, 0.0, 1.0));
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
