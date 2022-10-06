
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}
struct Resolution{
   inputwidth:f32,
   inputheight:f32,
   outputwidth:f32,
   outputheight:f32,
}


//!CONSTANT
//!VALUE INPUT_WIDTH
//let inputWidth : f32 = 1600.0;

//!CONSTANT
//!VALUE INPUT_HEIGHT
//let inputHeight : f32 = 900.0;

// //!CONSTANT
// //!VALUE INPUT_WIDTH
// let inputWidth : f32 = 1280.0;

// //!CONSTANT
// //!VALUE INPUT_HEIGHT
// let inputHeight : f32 = 720.0;

//!CONSTANT
//!VALUE OUTPUT_WIDTH
//let outputWidth : f32 = 1920.0;

//!CONSTANT
//!VALUE OUTPUT_HEIGHT
//let outputHeight : f32 = 1080.0;
@group(0) @binding(0)
var input: texture_2d<f32>;
@group(0) @binding(1)
var sam: sampler;

@group(1) @binding(0)// 1.
var<uniform> resolution:Resolution;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

fn min3(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->vec3<f32>{
    return min(a, min(b,c));
}
fn max3(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->vec3<f32>{
    return max(a, max(b,c));
}

struct FsrTap{
    aC:vec3<f32>,
    aW:f32,
}
struct FsrSet{
    dir:vec2<f32>,
    len:f32,
}
fn saturate(num:f32)->f32{
	  let a : f32 = 0.0;
	  let b:f32 = 1.0;
      //return max(min(num, max(a, b)),min(a, b));
	  return clamp(num,0.0,1.0);
}
fn matchSet(pp:vec2<f32>,biS:bool, biT:bool, biU:bool, biV:bool) -> f32{
	if(biS){
		return (1.0 - pp.x) * (1.0 - pp.y);
	}
	else if(biT){
        return pp.x * (1.0 - pp.y);
	}
	else if(biU){
       return (1.0 - pp.x) * pp.y;
	}
	else if(biV){
       return pp.x * pp.y;
	}
	else{
	   return 0.0;
	}
}

// Filtering for a given tap for the scalar.
fn FsrEasuTap(
	fsr:FsrTap, // Accumulated color, with negative lobe. // Accumulated weight.
	off:vec2<f32>, // Pixel offset from resolve position to tap.
	dir:vec2<f32>, // Gradient direction.
	len:vec2<f32>, // Length.
	lob:f32, // Negative lobe strength.
	clp:f32, // Clipping point.
	c:vec3<f32>  // Tap color.
)-> FsrTap {
	// Rotate offset by direction.
	var fsr1 : FsrTap = fsr;
    var aC : vec3<f32> = fsr1.aC;
    var aW : f32 = fsr1.aW;
	var v : vec2<f32> = vec2<f32>(0.0);
	v.x = (off.x * (dir.x)) + (off.y * dir.y);
	v.y = (off.x * (-dir.y)) + (off.y * dir.x);
	// Anisotropy.
	v = v * len;
	// Compute distance^2.
	var d2 : f32 = v.x * v.x + v.y * v.y;
	// Limit to the window as at corner, 2 taps can easily be outside.
	d2 = min(d2, clp);
	// Approximation of lanczos2 without sin() or rcp(), or sqrt() to get x.
	//  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
	//  |_______________________________________|   |_______________|
	//                   base                             window
	// The general form of the 'base' is,
	//  (a*(b*x^2-1)^2-(a-1))
	// Where 'a=1/(2*b-b^2)' and 'b' moves around the negative lobe.
	var wB : f32 = 2.0 / 5.0 * d2 - 1.0;
	var wA : f32 = lob * d2 - 1.0;
	wB = wB * wB;
	wA = wA * wA;
	wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
	var w : f32 = wB * wA;
	// Do weighted average.
	aC = aC + c * w;
    aW = aW + w;
    fsr1.aC = aC;
    fsr1.aW = aW;
    return fsr1;
}



// Accumulate direction and length.
fn FsrEasuSet(
	fsr : FsrSet,
	pp:vec2<f32>,
	biS:bool, biT:bool, biU:bool, biV:bool,
	lA:f32, lB:f32, lC:f32, lD:f32, lE:f32) -> FsrSet{
	// Compute bilinear weight, branches factor out as predicates are compiler time immediates.
	//  s t
	//  u v
	var fsr1 : FsrSet = fsr;
    var dir : vec2<f32> = fsr1.dir;
    var len : f32 = fsr1.len;
	var w : f32 = 0.0;
	w = matchSet(pp,biS,biT,biU,biV);
	// Direction is the '+' diff.
	//    a
	//  b c d
	//    e
	// Then takes magnitude from abs average of both sides of 'c'.
	// Length converts gradient reversal to 0, smoothly to non-reversal at 1, shaped, then adding horz and vert terms.
	var dc:f32 = lD - lC;
	var cb:f32 = lC - lB;
	var lenX:f32 = max(abs(dc), abs(cb));
	lenX = 1.0 / lenX;
	var dirX :f32 = lD - lB;
	dir.x = dir.x + (dirX * w);
	lenX = saturate(abs(dirX) * lenX);
	lenX = lenX * lenX;
	len = len + (lenX * w);
	// Repeat for the y axis.
	var ec :f32 = lE - lC;
	var ca : f32 = lC - lA;
	var lenY :f32 = max(abs(ec), abs(ca));
	lenY = 1.0 / lenY;
	var dirY:f32 = lE - lA;
	dir.y = dir.y + (dirY * w);
	lenY = saturate(abs(dirY) * lenY);
	lenY = lenY * lenY;
	len = len + (lenY * w);
    fsr1.dir = dir;
    fsr1.len = len;
    return fsr1;
}

fn gather_red_components(c: vec2<f32>) -> vec4<f32> {
   return textureGather(0,input,sam,c);
}
fn gather_green_components(c: vec2<f32>) -> vec4<f32> {
   return textureGather(1,input,sam,c);
}
fn gather_blue_components(c: vec2<f32>) -> vec4<f32> {
   return textureGather(2,input,sam,c);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
	var inputSize : vec2<f32>;
    inputSize.x = resolution.inputwidth;
    inputSize.y = resolution.inputheight;
	var outputSize : vec2<f32>;
    outputSize.x = resolution.outputwidth;
    outputSize.y = resolution.outputheight;

	//------------------------------------------------------------------------------------------------------------------------------
	  // Get position of 'f'.
	var pp:vec2<f32> = (floor(in.tex_coords * outputSize) + 0.5) / outputSize * inputSize - 0.5;
	var fp : vec2<f32> = floor(pp);
	pp = pp - fp;
	//------------------------------------------------------------------------------------------------------------------------------
	  // 12-tap kernel.
	  //    b c
	  //  e f g h
	  //  i j k l
	  //    n o
	  // Gather 4 ordering.
	  //  a b
	  //  r g
	  // For packed FP16, need either {rg} or {ab} so using the following setup for gather in all versions,
	  //    a b    <- unused (z)
	  //    r g
	  //  a b a b
	  //  r g r g
	  //    a b
	  //    r g    <- unused (z)
	  // Allowing dead-code removal to remove the 'z's  
	var p0 :vec2<f32> = fp + vec2<f32>(1.0, -1.0);
	// These are from p0 to avoid pulling two constants on pre-Navi hardware.
	var p1 : vec2<f32> = p0 + vec2<f32>(-1.0, 2.0);
	var p2: vec2<f32> = p0 + vec2<f32>(1.0, 2.0);
	var p3: vec2<f32> = p0 + vec2<f32>(0.0, 4.0);

	p0 = p0 / inputSize;
	p1 = p1 / inputSize;
	p2 = p2 / inputSize;
	p3 = p3 / inputSize;
    

	var bczzR : vec4<f32> = gather_red_components(p0);
	var bczzG : vec4<f32> = gather_green_components(p0);
	var bczzB : vec4<f32> = gather_blue_components(p0);
	var ijfeR : vec4<f32> = gather_red_components(p1);
	var ijfeG : vec4<f32> = gather_green_components(p1);
	var ijfeB : vec4<f32> = gather_blue_components(p1);
	var klhgR : vec4<f32> = gather_red_components(p2);
	var klhgG : vec4<f32> = gather_green_components(p2);
	var klhgB : vec4<f32> = gather_blue_components(p2);
	var zzonR : vec4<f32> = gather_red_components(p3);
	var zzonG : vec4<f32> = gather_green_components(p3);
	var zzonB : vec4<f32> = gather_blue_components(p3);
	//------------------------------------------------------------------------------------------------------------------------------
	  // Simplest multi-channel approximate luma possible (luma times 2, in 2 FMA/MAD).
	var bczzL : vec4<f32> = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
	var ijfeL : vec4<f32> = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
	var klhgL : vec4<f32> = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
	var zzonL : vec4<f32> = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
	// Rename.
	var bL :f32 = bczzL.x;
	var cL :f32 = bczzL.y;
	var iL :f32 = ijfeL.x;
	var jL :f32 = ijfeL.y;
	var fL :f32 = ijfeL.z;
	var eL :f32 = ijfeL.w;
	var kL :f32 = klhgL.x;
	var lL :f32 = klhgL.y;
	var hL :f32 = klhgL.z;
	var gL :f32 = klhgL.w;
	var oL :f32 = zzonL.z;
	var nL :f32 = zzonL.w;
	// Accumulate for bilinear interpolation.
    var fsr:FsrSet;
	fsr.dir = vec2<f32>(0.0,0.0);
	fsr.len = 0.0;
	fsr = FsrEasuSet(fsr, pp, true, false, false, false, bL, eL, fL, gL, jL);
	fsr = FsrEasuSet(fsr, pp, false, true, false, false, cL, fL, gL, hL, kL);
	fsr = FsrEasuSet(fsr, pp, false, false, true, false, fL, iL, jL, kL, nL);
    fsr = FsrEasuSet(fsr, pp, false, false, false, true, gL, jL, kL, lL, oL);
	//------------------------------------------------------------------------------------------------------------------------------
	  // Normalize with approximation, and cleanup close to zero.
	var dir2 :vec2<f32> = fsr.dir * fsr.dir;
	var dirR :f32 = dir2.x + dir2.y;
	var zro : bool = dirR < 1.0 / 32768.0;
	dirR = 1.0/(sqrt(dirR));
	if (zro) {dirR = 1.0;};
	if (zro) {fsr.dir.x = 1.0;};
	fsr.dir = fsr.dir * dirR;
	// Transform from {0 to 2} to {0 to 1} range, and shape with square.
	fsr.len = fsr.len * 0.5;
	fsr.len = fsr.len * fsr.len;
	// Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
	var stretch :f32 = (fsr.dir.x * fsr.dir.x + fsr.dir.y * fsr.dir.y) * 1.0/(max(abs(fsr.dir.x), abs(fsr.dir.y)));
	// Anisotropic length after rotation,
	//  x := 1.0 lerp to 'stretch' on edges
	//  y := 1.0 lerp to 2x on edges
	var len2:vec2<f32> = vec2<f32>(1.0 + (stretch - 1.0) * fsr.len, 1.0 - 0.5 * fsr.len );
	// Based on the amount of 'edge',
	// the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
	var lob : f32 = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * fsr.len;
	// Set distance^2 clipping point to the end of the adjustable window.
	var clp : f32 = 1.0/lob;
	//------------------------------------------------------------------------------------------------------------------------------
	  // Accumulation mixed with min/max of 4 nearest.
	  //    b c
	  //  e f g h
	  //  i j k l
	  //    n o
	var min4 = min(min3(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w), vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y)),
		vec3<f32>(klhgR.x, klhgG.x, klhgB.x));

	var max4:vec3<f32> = max(max3(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w), vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y)),
		vec3<f32>(klhgR.x, klhgG.x, klhgB.x));
	// Accumulation.
	var fsr2:FsrTap;
	fsr2.aC = vec3<f32>(0.0,0.0,0.0);
	fsr2.aW =0.0;
	fsr2=FsrEasuTap(fsr2, vec2<f32>(0.0, -1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(bczzR.x, bczzG.x, bczzB.x)); // b
	fsr2=FsrEasuTap(fsr2, vec2<f32>(1.0, -1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(bczzR.y, bczzG.y, bczzB.y)); // c
	fsr2=FsrEasuTap(fsr2, vec2<f32>(-1.0, 1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(ijfeR.x, ijfeG.x, ijfeB.x)); // i
	fsr2=FsrEasuTap(fsr2, vec2<f32>(0.0, 1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y)); // j
	fsr2=FsrEasuTap(fsr2, vec2<f32>(0.0, 0.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z)); // f
	fsr2=FsrEasuTap(fsr2, vec2<f32>(-1.0, 0.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(ijfeR.w, ijfeG.w, ijfeB.w)); // e
	fsr2=FsrEasuTap(fsr2, vec2<f32>(1.0, 1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(klhgR.x, klhgG.x, klhgB.x)); // k
	fsr2=FsrEasuTap(fsr2, vec2<f32>(2.0, 1.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(klhgR.y, klhgG.y, klhgB.y)); // l
	fsr2=FsrEasuTap(fsr2, vec2<f32>(2.0, 0.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(klhgR.z, klhgG.z, klhgB.z)); // h
	fsr2=FsrEasuTap(fsr2, vec2<f32>(1.0, 0.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(klhgR.w, klhgG.w, klhgB.w)); // g
	fsr2=FsrEasuTap(fsr2, vec2<f32>(1.0, 2.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(zzonR.z, zzonG.z, zzonB.z)); // o
	fsr2=FsrEasuTap(fsr2, vec2<f32>(0.0, 2.0) - pp, fsr.dir, len2, lob, clp, vec3<f32>(zzonR.w, zzonG.w, zzonB.w)); // n
  //------------------------------------------------------------------------------------------------------------------------------
	// Normalize and dering.
	var c : vec3<f32> = min(max4, max(min4, fsr2.aC * (1.0/fsr2.aW)));
    
	return vec4<f32>(c, 1.0);
}

