

let sharpness : f32 = 0.15;

struct CameraUniform {
    view_proj: mat4x4<f32>,
}
struct Resolution{
   inputwidth:f32,
   inputheight:f32,
   outputwidth:f32,
   outputheight:f32,
}


@group(0)@binding(0)
var input: texture_2d<f32>;
@group(0)@binding(1)
var sam: sampler;
@group(1)@binding(0)// 1.
var<uniform> resolution:Resolution;
@group(2)@binding(0) // 1.
var<uniform> camera: CameraUniform;

// struct CameraUniform {
//     view_proj: mat4x4<f32>;
// };
// [[group(1),binding(0)]]// 1.
// var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
   @location(0)tex_coords: vec2<f32>,
}

fn min3f(a:f32,b:f32,c:f32)->f32{
    return min(a, min(b,c));
}
fn max3f(a:f32,b:f32,c:f32)->f32{
    return max(a, max(b,c));
}

fn saturate(num:f32)->f32{
	  let a : f32 = 0.0;
	  let b:f32 = 1.0;
      //return max(min(num, max(a, b)),min(a, b));
	  return clamp(num,0.0,1.0);
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
	//var sp : vec2<i32> = vec2<i32>(floor(in.tex_coords * vec2<f32>(inputWidthRcas, inputHeightRcas)));
    let FSR_RCAS_LIMIT :f32 = 0.25-(1.0/16.0);
	// Algorithm uses minimal 3x3 pixel neighborhood.
	//    b 
	//  d e f
	//    h
	// var b : vec3<f32>  = textureLoad(input,vec2<i32>(sp.x, sp.y - 1), 0).rgb;
	// var d : vec3<f32>  = textureLoad(input,vec2<i32>(sp.x - 1, sp.y), 0).rgb;
	// let e : vec3<f32>  = textureLoad(input,vec2<i32>(sp), 0).rgb;
	// var f : vec3<f32>  = textureLoad(input,vec2<i32>(sp.x + 1, sp.y), 0).rgb;
	// var h : vec3<f32>  = textureLoad(input,vec2<i32>(sp.x, sp.y + 1), 0).rgb;

	let b : vec3<f32> = textureSample(input,sam,in.tex_coords + vec2<f32>(0.0, -resolution.outputheight)).rgb;
	let d : vec3<f32> = textureSample(input,sam,in.tex_coords + vec2<f32>(-resolution.outputwidth, 0.0)).rgb;
	var e : vec3<f32> = textureSample(input,sam,in.tex_coords).rgb;
	let f : vec3<f32> = textureSample(input,sam,in.tex_coords + vec2<f32>(resolution.outputwidth, 0.0)).rgb;
	let h : vec3<f32> = textureSample(input,sam,in.tex_coords + vec2<f32>(0.0, resolution.outputheight)).rgb;
	// Rename (32-bit) or regroup (16-bit).
	var bR :f32 = b.r;
	var bG :f32  = b.g;
	var bB :f32  = b.b;
	var dR :f32  = d.r;
	var dG :f32  = d.g;
	var dB :f32  = d.b;
	var eR :f32  = e.r;
	var eG :f32  = e.g;
	var eB :f32  = e.b;
	var fR :f32 = f.r;
	var fG :f32  = f.g;
	var fB :f32  = f.b;
	var hR :f32 = h.r;
	var hG :f32  = h.g;
	var hB:f32  = h.b;

	var nz:f32 = 0.0;

	// Luma times 2.
	var bL :f32  = bB * 0.5 + (bR * 0.5 + bG);
	var dL:f32  = dB * 0.5 + (dR * 0.5 + dG);
	var eL:f32  = eB * 0.5 + (eR * 0.5 + eG);
	var fL:f32  = fB * 0.5 + (fR * 0.5 + fG);
	var hL:f32  = hB * 0.5 + (hR * 0.5 + hG);

	// Noise detection.
	nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
	nz = saturate(abs(nz) * 1.0/(max3f(max3f(bL, dL, eL), fL, hL) - min3f(min3f(bL, dL, eL), fL, hL)));
	nz = -0.5 * nz + 1.0;

	// Min and max of ring.
	var mn4R :f32 =  min(min3f(bR, dR, fR), hR);
	var mn4G :f32  = min(min3f(bG, dG, fG), hG);
	var mn4B :f32  = min(min3f(bB, dB, fB), hB);
	var mx4R :f32  = max(max3f(bR, dR, fR), hR);
	var mx4G :f32  = max(max3f(bG, dG, fG), hG);
	var mx4B :f32  = max(max3f(bB, dB, fB), hB);
	// Immediate constants for peak range.
	var peakC :vec2<f32> = vec2<f32>( 1.0, -1.0 * 4.0 );
	// Limiters, these need to be high precision RCPs.
	var hitMinR :f32  = min(mn4R, eR) * 1.0/(4.0 * mx4R);
	var hitMinG :f32 = min(mn4G, eG) * 1.0/(4.0 * mx4G);
	var hitMinB :f32  = min(mn4B, eB) * 1.0/(4.0 * mx4B);
	var hitMaxR :f32 = (peakC.x - max(mx4R, eR)) * 1.0/(4.0 * mn4R + peakC.y);
	var hitMaxG :f32  = (peakC.x - max(mx4G, eG)) * 1.0/(4.0 * mn4G + peakC.y);
	var hitMaxB :f32 = (peakC.x - max(mx4B, eB)) * 1.0/(4.0 * mn4B + peakC.y);
	var lobeR:f32  = max(-hitMinR, hitMaxR);
	var lobeG:f32  = max(-hitMinG, hitMaxG);
	var lobeB :f32  = max(-hitMinB, hitMaxB);
	var lobe :f32  = max(-FSR_RCAS_LIMIT, min(max3f(lobeR, lobeG, lobeB), 0.0)) * sharpness;

	// Apply noise removal.
	lobe = lobe * nz;

	// Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
	var rcpL :f32  = 1.0/(4.0 * lobe + 1.0);
	var c:vec3<f32>  = vec3<f32>(
		(lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) * rcpL,
		(lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) * rcpL,
		(lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) * rcpL
	);
	return vec4<f32>(c, 1.0);
}