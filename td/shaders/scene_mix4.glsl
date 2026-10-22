// GLSL TOP 'scene_mix4': weighted sum of up to four scenes (particles, moon, asteroid, ghost). Weights glide between effects
// (300 ms) so changing finger cross-fades instead of cutting. Vectors: uW = (w_particles, w_moon, w_asteroid, w_ghost)
uniform vec4 uW;

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;
    vec3 c = texture(sTD2DInputs[0], uv).rgb * uW.x + texture(sTD2DInputs[1], uv).rgb * uW.y
           + texture(sTD2DInputs[2], uv).rgb * uW.z + texture(sTD2DInputs[3], uv).rgb * uW.w;
    fragColor = TDOutputSwizzle(vec4(c, 1.0));
}
