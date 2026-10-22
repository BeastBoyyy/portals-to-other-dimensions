// GLSL TOP 'scene_blend': 300 ms cross-fade between scene A (input 0) and scene B (input 1).
// Vectors: uMix = (mix, 0, 0, 0)  -- mix comes from CHOP channel scene_mix (already slewed in presenter_chop)
uniform vec4 uMix;

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;
    vec3 a = texture(sTD2DInputs[0], uv).rgb;
    vec3 b = texture(sTD2DInputs[1], uv).rgb;
    fragColor = TDOutputSwizzle(vec4(mix(a, b, clamp(uMix.x, 0.0, 1.0)), 1.0));
}
