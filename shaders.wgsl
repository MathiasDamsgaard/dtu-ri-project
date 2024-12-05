struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct FSOut {
    @location(0) frame: vec4f,
    @location(1) accum: vec4f,
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    //color: vec3f,
    shader: u32,
    diffuse: vec3f,
    ambient: vec3f,
    texcoords: vec2f,
    emit: bool,
    factor: vec3f,
    ext_coeff: vec3f
};

struct Light {
    l_i: vec3f,
    w_i: vec3f,
    dist: f32
};

struct Uniforms_f {
    aspect: f32,
    gamma: f32,
    l_cam: f32,
    zd_cam: f32,
    cam_const: f32,
    l_proj: f32,
    zd_proj: f32,
    proj_const: f32
};

struct Uniforms_ui {
    shader_1: u32,
    shader_2: u32,
    width: u32,
    height: u32,
    frame: u32,
    texture_width: u32,
    texture_height: u32,
    dir_light: u32,
    proj_light: u32,
    indir_light: u32
};

struct Onb {
    tangent: vec3f,
    binormal: vec3f,
    normal: vec3f,
};

struct Aabb {
    min: vec3f,
    max: vec3f,
};

struct Material {
    color: vec4f,
    emission: vec4f,
    specular: vec4f,
    illum: vec4f
}

struct Attribute {
    position: vec4f,
    normal: vec4f,
};


@group(0) @binding(0) var<uniform> uniforms_f: Uniforms_f;
@group(0) @binding(1) var<uniform> uniforms_ui: Uniforms_ui;

@group(0) @binding(2) var<storage> attributes: array<Attribute>;
@group(0) @binding(3) var<storage> meshFaces: array<vec4u>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var<storage> lightIndices: array<u32>;

@group(0) @binding(6) var<uniform> aabb: Aabb;
@group(0) @binding(7) var<storage> treeIds: array<u32>;
@group(0) @binding(8) var<storage> bspTree: array<vec4u>;
@group(0) @binding(9) var<storage> bspPlanes: array<f32>;

@group(0) @binding(10) var renderTexture: texture_2d<f32>;

@group(0) @binding(11) var projectorTexture: texture_2d<f32>;

const PI = 3.14159265359;
const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;
var<private> branch_node: array<vec2u, MAX_LEVEL>;
var<private> branch_ray: array<vec2f, MAX_LEVEL>;


@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

fn sample_point_in_circle(t: ptr<function, u32>, radius: f32) -> vec2f {
    let r = sqrt(rnd(t)) * (radius); // sqrt to ensure uniform distribution in the circle
    let theta = 2.0 * PI * rnd(t);

    // Convert to cartesian coordinates
    let x = r * cos(theta);
    let y = r * sin(theta);

    return vec2f(x, y);
}

fn get_camera_ray(ipcoords: vec2f, t: ptr<function, u32>) -> Ray {
    // Scene description
    // const e = vec3f(277.0, 275.0, -570.0); // eye point
    // const p = vec3f(277.0, 275.0, 0.0); // look-at point (view point)
    // const u = vec3f(0.0, 1.0, 0.0); // up-vector (up direction)
    // const e = vec3f(0.0, 10.0, 0.0001); // eye point
    const e = vec3f(0.0, 3.0, 3.0); // eye point
    const p = vec3f(0.0, 0.0, 0.0); // look-at point (view point)
    const u = vec3f(0.0, 1.0, 0.0); // up-vector (up direction)
    let d = uniforms_f.cam_const; // camera constant
    let z_d = uniforms_f.zd_cam; // distance to the focal plane
    let l = uniforms_f.l_cam; // lens radius
    let f = 1.0 / (1.0 / d + 1.0 / z_d); // focal length
    let delta = f * l / (z_d - f); // diameter of circle of confusion

    // Compute camera coordinate system (WGSL has vector operations like normalize and cross)
    let v = normalize(p - e);
    let b1 = normalize(cross(v, u));
    let b2 = cross(b1, v);
    let w = normalize(ipcoords.x*b1 + ipcoords.y*b2 + d*v);

    let fp = e + z_d * w; // focal point

    let xy = sample_point_in_circle(t, delta/2.0);

    // Transform to the vector space of the camera
    let offset = xy.x * b1 + xy.y * b2;

    // Implement ray generation
    var ray: Ray;
    ray.origin = e + offset;
    ray.direction = normalize(fp - ray.origin); // direction vector from the eye to the focal point
    ray.tmin = 0.0;
    ray.tmax = 1.0e16;
    return ray;
}

fn get_plane_intersection(
    r_origin: vec3f,
    r_dir: vec3f,
    p_pos:vec3f,
    p_norm: vec3f,
    intersection: ptr<function, vec3f>
) -> bool {
    var denominator = dot(r_dir, p_norm);
    var epsilon = 0.00001;

    if (abs(denominator) <= epsilon) {
        // the line is parallel to the plane
        return false;
    }

    var tp = dot(p_pos - r_origin, p_norm) / denominator;

    if(dot(p_norm, r_dir) > 0.0) {
        // the plane is facing the ray
        return false;
    }

    *intersection = r_origin + tp*r_dir;
    return true;
}


fn sample_ideal_projector(pos: vec3f, t: ptr<function, u32>) -> Light {
    const position = vec3f(0.0, 8.0, 0.0);
    const updir = vec3f(0.0, 0.0, 1.0);
    const lookAt = vec3f(0.0, 0.0, 0.0);
    var dist = length(position - pos);

    let up = normalize(updir - dot(updir, lookAt)*lookAt);

    let plane_norm = normalize(lookAt - position);
    let right = normalize(cross(plane_norm, up));

    let v = normalize(position - pos);

    let d = uniforms_f.proj_const;
    let z_d = uniforms_f.zd_proj;
    let l = uniforms_f.l_proj;


    let f = 1.0 / (1.0 / d + 1.0 / z_d);

    let z = length(position - pos);
    let delta = f * l * abs((z_d/z - 1)/(z_d - f));

    let xy = sample_point_in_circle(t, delta/2.0);

    var intersection = vec3f(0.0);

    let plane_real_pos = position + normalize(plane_norm)*d;

    let has_intersection = get_plane_intersection(pos, v, plane_real_pos, plane_norm, &intersection);

    if(!has_intersection) {
        return Light(vec3f(0.0), normalize(position-pos), dist);
    }

    let proj = intersection - plane_real_pos;
    var x = dot(proj, right);
    var y = dot(proj, up);

    x += xy.x;
    y += xy.y;

    let aspect_ratio = f32(uniforms_ui.texture_width) / f32(uniforms_ui.texture_height);

    let scaled_y = y * aspect_ratio;

    if(x < -1.0 || x > 1.0 || scaled_y < -1.0 || scaled_y > 1.0) {
        return Light(vec3f(0.0), normalize(position-pos), dist);
    }

    let xs = (-x + 1.0) / 2.0;
    let ys = (-scaled_y + 1.0) / 2.0;


    // multiply by texture size and cast to int
    let ut = xs * f32(uniforms_ui.texture_width);
    let vt = ys * f32(uniforms_ui.texture_height);


    let col = textureLoad(projectorTexture, vec2i(i32(ut), i32(vt)), 0).rgb;

    let intensity = vec3f(col * PI*20);
    var l_i = intensity / pow(length(position - pos), 2);
    var w_i = normalize(position - pos);
    return Light(l_i, w_i, dist);
}


fn sample_point_light(pos: vec3f) -> Light {
    const position = vec3f(0.0, 1.0, 0.0);
    const intensity = vec3f(PI, PI, PI);
    var l_i = intensity / pow(length(position - pos), 2);
    var w_i = normalize(position - pos);
    var dist = length(position - pos);
    return Light(l_i, w_i, dist);
}

fn sample_direction_light() -> Light {
    const emission = vec3f(0.1);
    const direction = normalize(vec3f(-1.0));
    var dist = 1.0e16;
    return Light(emission, -direction, dist);
}

// Previously sample_emissive_triangle
fn sample_area_light(hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> Light {
    // Sample a random triangle light source
    var n = f32(arrayLength(&lightIndices));
    var idx = u32(rnd(t) * n);

    let v0 = attributes[meshFaces[lightIndices[idx]].x].position.xyz;
    let v1 = attributes[meshFaces[lightIndices[idx]].y].position.xyz;
    let v2 = attributes[meshFaces[lightIndices[idx]].z].position.xyz;

    let n0 = attributes[meshFaces[lightIndices[idx]].x].normal.xyz;
    let n1 = attributes[meshFaces[lightIndices[idx]].y].normal.xyz;
    let n2 = attributes[meshFaces[lightIndices[idx]].z].normal.xyz;

    let light_emission = materials[meshFaces[lightIndices[idx]].w].emission.rgb;

    // Sample a random point on the light source
    var rnd_1 = rnd(t); var rnd_2 = rnd(t);

    var alpha = 1 - sqrt(rnd_1);
    var beta = (1 - rnd_2) * sqrt(rnd_1);
    var gamma = rnd_2 * sqrt(rnd_1);

    // Interpolated barycentric coordinates and normal
    let light_pos = alpha * v0 + beta * v1 + gamma * v2;
    let light_normal = normalize(alpha * n0 + beta * n1 + gamma * n2);
    let light_area = 0.5 * length(cross(v1 - v0, v2 - v1));

    let light_dir = light_pos - hit.position;
    let light_dist = length(light_dir);
    let w_i = normalize(light_dir);

    let cos_theta_l = max(dot(-w_i, light_normal), 0.0);
    let L_i = light_emission * light_area * cos_theta_l * n / pow(light_dist, 2);

    return Light(L_i, w_i, light_dist);
}

fn sample_cosine_hemisphere(t: ptr<function, u32>, n: vec3f) -> vec3f {
    var rnd_1 = rnd(t); var rnd_2 = rnd(t);

    var theta = acos(sqrt(1 - rnd_1));
    var phi = 2.0 * PI * rnd_2;

    var tangent_dir = spherical_direction(sin(theta), cos(theta), phi);
    var dir = rotate_to_normal(n, tangent_dir);
    return dir;
}

// Given spherical coordinates, where theta is the polar angle and phi is the
// azimuthal angle, this function returns the corresponding direction vector
fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> vec3f {
    return vec3f(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

// Given a direction vector v sampled around the z-axis of a local coordinate system,
// this function applies the same rotation to v as is needed to rotate the z-axis to
// the actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
// Duff et al., Journal of Computer Graphics Techniques 6, 2017].
fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
    let s = sign(n.z + 1.0e-16f);
    let a = -1.0f/(1.0f + abs(n.z));
    let b = n.x*n.y*a;
    return vec3f(1.0f + n.x*n.x*a, b, -s*n.x)*v.x + vec3f(s*b, s*(1.0f + n.y*n.y*a), -n.y)*v.y + n*v.z;
}

// PRNG xorshift seed generator by NVIDIA
fn tea(val0: u32, val1: u32) -> u32 {
    const N = 16u; // User specified number of iterations
    var v0 = val0; var v1 = val1; var s0 = 0u;
    for(var n = 0u; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    return v0;
}

// Generate random unsigned int in [0, 2^31)
fn mcg31(prev: ptr<function, u32>) -> u32 {
    const LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    *prev = (LCG_A * (*prev)) & 0x7FFFFFFF;
    return *prev;
}

// Generate random float in [0, 1)
fn rnd(prev: ptr<function, u32>) -> f32 {
    return f32(mcg31(prev)) / f32(0x80000000);
}

//  Fresnel reflectanc
fn fresnel_R(ni: f32, nt: f32, cos_theta_i: f32, cos_theta_t: f32) -> f32 {
    if (cos_theta_t < 0.0) {return 1.0;}

    var r_perpendicular = (ni*cos_theta_i - nt*cos_theta_t) / (ni*cos_theta_i + nt*cos_theta_t);
    var r_parallel = (nt*cos_theta_i - ni*cos_theta_t) / (nt*cos_theta_i + ni*cos_theta_t);
    return 0.5 * (r_parallel*r_parallel + r_perpendicular*r_perpendicular);
}


fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, normal: vec3f, plane_onb: Onb) -> bool {
    // Check denominator to avoid division by zero
    var denominator = dot(r.direction, normal);
    var epsilon = 0.00001;
    if (abs(denominator) <= epsilon) {
        return false;
    }

    var tp = dot(position - r.origin, normal) / denominator;

    if (tp > r.tmin && tp < r.tmax) {
        hit.has_hit = true;
        hit.dist = tp;
        hit.position = r.origin + tp*r.direction;

        // Texture uv coordinates
        var u = dot(hit.position - 0, plane_onb.tangent);
        var v = dot(hit.position - 0, plane_onb.binormal);
        var texture_scaling = 0.2/10;
        hit.texcoords = vec2f(u, v)*texture_scaling;

        if (dot(normal, r.direction) > 0.0) {
            hit.normal = -normalize(normal);
        } else {
            hit.normal = normalize(normal);
        }
        return true;
    }
    return false;
}

fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, v_idx: u32) -> bool {
    // Get the vertices of the triangle
    let v0 = meshFaces[v_idx].x;
    let v1 = meshFaces[v_idx].y;
    let v2 = meshFaces[v_idx].z;

    // Get the material of the triangle and set the hit info
    let tri_material = materials[meshFaces[v_idx].w];

    var v : array<vec3f, 3> = array(
        attributes[v0].position.xyz,
        attributes[v1].position.xyz,
        attributes[v2].position.xyz
    );
    // Using .xyz to convert vec4f to vec3f
    var normals : array<vec3f, 3> = array(
        attributes[v0].normal.xyz,
        attributes[v1].normal.xyz,
        attributes[v2].normal.xyz
    );

    var e0 = v[1] - v[0]; var e1 = v[2] - v[0]; var n = cross(e0, e1);
    var denominator = dot(r.direction, n);
    var beta = dot(cross(v[0]- r.origin, r.direction), e1) / denominator;
    var gamma = -dot(cross(v[0] - r.origin, r.direction), e0) / denominator;
    var alpha = 1.0 - beta - gamma;

    if (alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0) {
        var tp = dot(v[0] - r.origin, n) / denominator;

        if (tp >= r.tmin && tp <= r.tmax) {
            hit.has_hit = true;
            hit.dist = tp;
            hit.position = r.origin + tp*r.direction;

            var interpolated_normal = alpha*normals[0] + beta*normals[1] + gamma*normals[2];
            if (dot(n, r.direction) > 0.0) {
                hit.normal = -normalize(interpolated_normal);
            } else {
                hit.normal = normalize(interpolated_normal);
            }

            hit.ambient = tri_material.emission.rgb;
            hit.diffuse = tri_material.color.rgb;
            hit.shader = uniforms_ui.shader_2;

            return true;
        }
    }
    return false;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32) -> bool {
    var halfb = dot(r.origin - center, r.direction);
    var c = dot(r.origin - center, r.origin - center) - pow(radius, 2);
    var determinant = pow(halfb, 2) - c;
    if (determinant < 0.0) { return false; }

    var t1 = -halfb - sqrt(determinant);

    if (t1 >= r.tmin && t1 <= r.tmax) {
        hit.has_hit = true;
        hit.dist = t1;
        hit.position = r.origin + t1*r.direction;
        hit.normal = normalize(hit.position - center);
        return true;
    }

    var t2 = -halfb + sqrt(determinant);

    if (t2 >= r.tmin && t2 <= r.tmax) {
        hit.has_hit = true;
        hit.dist = t2;
        hit.position = r.origin + t2*r.direction;
        hit.normal = normalize(hit.position - center);
        return true;
    }
    return false;
}

fn intersect_min_max(r: ptr<function, Ray>) -> bool {
    let p1 = (aabb.min - r.origin)/r.direction;
    let p2 = (aabb.max - r.origin)/r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let tmin = max(pmin.x, max(pmin.y, pmin.z));
    let tmax = min(pmax.x, min(pmax.y, pmax.z));
    if(tmin > tmax || tmin > r.tmax || tmax < r.tmin) { return false; }
    r.tmin = max(tmin - 1.0e-3f, r.tmin);
    r.tmax = min(tmax + 1.0e-3f, r.tmax);
    return true;
}

fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;
    for(var i = 0u; i <= MAX_LEVEL; i++) {
        let tree_node = bspTree[node];
        let node_axis_leaf = tree_node.x&3u;
        if(node_axis_leaf == BSP_LEAF) {
        // A leaf was found
        let node_count = tree_node.x >> 2u;
        let node_id = tree_node.y;
        var found = false;
        for (var j=0u; j < node_count; j++) {
            let obj_idx = treeIds[node_id + j];
            if (intersect_triangle(*r, hit, obj_idx)) {
                r.tmax = hit.dist;
                found = true;
            }
        }
        if (found) { return true; }
        else if (branch_lvl == 0u) { return false; }
        else {
            branch_lvl--;
            i = branch_node[branch_lvl].x;
            node = branch_node[branch_lvl].y;
            r.tmin = branch_ray[branch_lvl].x;
            r.tmax = branch_ray[branch_lvl].y;
            continue;
        }
        }
        let axis_direction = r.direction[node_axis_leaf];
        let axis_origin = r.origin[node_axis_leaf];
        if(axis_direction >= 0.0f) {
            near_node = tree_node.z; // left
            far_node = tree_node.w; // right
        }
        else {
            near_node = tree_node.w; // right
            far_node = tree_node.z; // left
        }
        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin)/denom;
        if(t > r.tmax) { node = near_node; }
        else if(t < r.tmin) { node = far_node; }
        else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = r.tmax;
            branch_lvl++;
            r.tmax = t;
            node = near_node;
        }
    }
    return false;
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    var result = vec3f(0.0);

    if (hit.emit) {
        result += hit.ambient;
    }

    var dirlight = sample_direction_light();
    var plight = sample_point_light(hit.position);
    // var area = sample_area_light(hit, t);
    var proj = sample_ideal_projector(hit.position, t);

    if(uniforms_ui.dir_light == 1u) {
        var light = dirlight;
        var epsilon = 1e-1;
        var shadow_ray = Ray(hit.position, light.w_i, epsilon, light.dist - epsilon);
        var shadow_hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), 0, vec3f(0.0), vec3f(0.0), vec2f(0.0), false, vec3f(1.0), vec3f(0.0));

        if (!intersect_scene(&shadow_ray, &shadow_hit)) {
            // compute lambertian shading
            var l_o = hit.diffuse / PI * light.l_i * max(0.0, dot(hit.normal, light.w_i));
            result += l_o;
        }
    }

    if(uniforms_ui.proj_light == 1u) {
        var light = proj;
        var epsilon = 1e-1;
        var shadow_ray = Ray(hit.position, light.w_i, epsilon, light.dist - epsilon);
        var shadow_hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), 0, vec3f(0.0), vec3f(0.0), vec2f(0.0), false, vec3f(1.0), vec3f(0.0));

        if (!intersect_scene(&shadow_ray, &shadow_hit)) {
            // compute lambertian shading
            var l_o = hit.diffuse / PI * light.l_i * max(0.0, dot(hit.normal, light.w_i));
            result += l_o;
        }
    }

    // Indirect illumination

    // Probability of diffuse reflection
    if (uniforms_ui.indir_light == 1u) {
        var temp = hit.factor * hit.diffuse;
        var P_d = (temp.r + temp.g + temp.b) / 3.0;
        var xi = rnd(t);

        if (xi < P_d) {
            var indirect_dir = sample_cosine_hemisphere(t, hit.normal);

            // This ray will be used in the next iteration of raytrace
            (*r).origin = hit.position;
            (*r).direction = indirect_dir;
            (*r).tmin = 1e-3;
            (*r).tmax = 1e16;

            hit.has_hit = false;  // Continue tracing
            hit.emit = false; // No longer emitting after first bounce
            hit.factor = temp / P_d;
        } else {
            hit.has_hit = true;
        }
    }
    
    return result;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    var light = sample_point_light(hit.position);

    // shadow and check if there is anything between the hit and light source
    var shadow_ray = Ray(hit.position, light.w_i, 1e-4, light.dist);
    var shadow_hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), 0, vec3f(0.0), vec3f(0.0), vec2f(0.0), false, vec3f(0.0), vec3f(0.0));
    if (intersect_scene(&shadow_ray, &shadow_hit)) { return hit.ambient; }

    const s = 42; // sphere shinniness
    const rho_s = 0.1; // specular reflectance
    var rho_d = hit.diffuse; // diffuse reflectance

    // phong equation
    var l_r = (rho_d/PI + rho_s * (s+2)/(2*PI) * pow(dot(light.w_i, reflect(-(-r.direction), hit.normal)), s) *
              light.l_i * dot(light.w_i, hit.normal));
    return l_r + hit.ambient;
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // compute new direction for the ray
    (*r).origin = hit.position;
    (*r).direction = reflect(-(-r.direction), hit.normal);
    (*r).tmin = 1e-2;
    (*r).tmax = 1e16;

    // want to run another iteration from the hit ray
    hit.has_hit = false;
    hit.emit = true;
    return vec3f(0.0);
}

fn refraction(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    var index = 1.5; // refractive index
    var n_t = 0.0; var n_i = 0.0;

    if (dot(hit.normal, -r.direction) > 0.0) {
        n_t = index; n_i = 1.0;
    } else {
        n_t = 1.0; n_i = index;
        hit.normal = -hit.normal;
    }

    var cos_theta_squared = 1 - pow(n_i/n_t, 2) * (1 - pow(dot(hit.normal, -r.direction), 2));

    if (cos_theta_squared < 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var w_t = n_i/n_t * (dot(-r.direction, hit.normal) * hit.normal - (-r.direction)) - hit.normal * sqrt(cos_theta_squared);
    (*r).origin = hit.position;
    (*r).direction = w_t;
    (*r).tmin = 1e-4;
    (*r).tmax = 1e16;

    hit.has_hit = false; // we need to keep the ray going
    return vec3f(0.0);
}

fn glossy(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    var l_r = phong(r, hit);
    var res = refraction(r, hit);
    return l_r + res;
}

fn transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    var index = 1.5; // refractive index
    var n_t = 0.0; var n_i = 0.0;
    var transmit_prob = 1.0;
    var beam_transmittance = exp(-hit.ext_coeff * hit.dist);

    if (dot(hit.normal, -r.direction) > 0.0) {
        n_t = index; n_i = 1.0;
    } else {
        n_t = 1.0; n_i = index;
        hit.normal = -hit.normal;

        // Inside the object
        transmit_prob = (beam_transmittance.x + beam_transmittance.y + beam_transmittance.z) / 3.0;
    }

    var xi = rnd(t);
    if (xi < transmit_prob) {
        // transmission
        var cos_theta_i = dot(hit.normal, -r.direction);
        var cos_theta_t = sqrt(1 - pow(n_i/n_t, 2) * (1 - pow(cos_theta_i, 2)));

        var R = fresnel_R(n_i, n_t, cos_theta_i, cos_theta_t);

        var xi = rnd(t);
        if (xi < R) {
            // reflection
            var reflection_direction = reflect(-(-r.direction), hit.normal);
            (*r).direction = reflection_direction;
        } else {
            // refraction
            var wt = (n_i / n_t) * (cos_theta_i * hit.normal - (-r.direction)) - hit.normal * cos_theta_t;
            (*r).direction = wt;
        }

        (*r).origin = hit.position;
        (*r).tmin = 1e-1;
        (*r).tmax = 1e16;

        hit.has_hit = false;
        hit.emit = true;
        if (transmit_prob < 0.99999) {
            hit.factor *= beam_transmittance / transmit_prob;
        }
    }
    else {
        // absorption
        hit.has_hit = true;
    }

    return vec3f(0.0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    switch (*hit).shader {
        case 1 { return lambertian(r, hit, t); }
        case 2 { return phong(r, hit); }
        case 3 { return mirror(r, hit); }
        case 4 { return refraction(r, hit); }
        case 5 { return glossy(r, hit); }
        case 6 { return transparent(r, hit, t); }
        case default { return (*hit).diffuse + (*hit).ambient; }
    }
}

// fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool{
//     if (!intersect_min_max(r)) { return false; }

//     const center_s_left = vec3f(420.0, 90.0, 370.0);
//     const radius_s_left = 90.0;
//     if (intersect_sphere(*r, hit, center_s_left, radius_s_left)) {
//         hit.shader = uniforms_ui.shader_1;
//         hit.ext_coeff = vec3f(0.0);
//         r.tmax = hit.dist;
//     }

//     const center_s_right = vec3f(130.0, 90.0, 250.0);
//     const radius_s_right = 90.0;
//     if (intersect_sphere(*r, hit, center_s_right, radius_s_right)) {
//         hit.shader = uniforms_ui.shader_2;
//         hit.ext_coeff = vec3f(1e-3, 0.9, 1e-3);
//         r.tmax = hit.dist;
//     }

//     if (intersect_trimesh(r, hit)) {
//         hit.ext_coeff = vec3f(0.0);
//         r.tmax = hit.dist;
//     }

//     return (*hit).has_hit;
// }

fn intersect_scene(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    if(intersect_sphere(*ray, hit, vec3f(0.0, 0.5, 0.0), 0.5)) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.6, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.6, 0.1);
        (*hit).shader = uniforms_ui.shader_1;
        (*hit).emit = false;
        return true;
    }

    if(intersect_sphere(*ray, hit, vec3f(1.5, 0.5, 0.0), 0.5)) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.6, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.6, 0.1);
        (*hit).shader = uniforms_ui.shader_2;
        (*hit).emit = false;
        return true;
    }

    if(intersect_sphere(*ray, hit, vec3f(-1.5, 0.5, 0.0), 0.5)) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.6, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.6, 0.1);
        (*hit).shader = uniforms_ui.shader_2;
        (*hit).emit = false;
        return true;
    }

    if(intersect_sphere(*ray, hit, vec3f(0.0, 0.5, 1.5), 0.5)) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.6, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.6, 0.1);
        (*hit).shader = uniforms_ui.shader_1;
        (*hit).emit = false;
        return true;
    }

    if(intersect_sphere(*ray, hit, vec3f(0.0, 0.5, -1.5), 0.5)) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.6, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.6, 0.1);
        (*hit).shader = uniforms_ui.shader_1;
        (*hit).emit = false;
        return true;
    }

    if(intersect_plane(*ray, hit, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), Onb(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0)))) {
        (*ray).tmax = (*hit).dist;
        (*hit).ambient = vec3f(0.2, 0.1, 0.1);
        (*hit).diffuse = vec3f(0.2, 0.1, 0.1);
        (*hit).shader = 1; // lambertian
        (*hit).emit = false;
        return true;
    }

    return false;
}

fn texture_nearest(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let res = textureDimensions(texture);
    let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
    let ab = st*vec2f(res);
    let UV = vec2u(ab + 0.5) % res;
    let texcolor = textureLoad(texture, UV, 0);
    return texcolor.rgb;
}

fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let res = textureDimensions(texture);
    let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
    let ab = st*vec2f(res);

    // Calculate the integer and fractional parts of the texture coordinates
    let ab_floor = floor(ab);
    let ab_frac = ab - ab_floor;

    // Get the four nearest texels around the fractional coordinate
    let p0 = vec2u(ab_floor); // Top-left texel
    let p1 = (p0 + vec2u(1, 0)) % res; // Top-right texel
    let p2 = (p0 + vec2u(0, 1)) % res; // Bottom-left texel
    let p3 = (p0 + vec2u(1, 1)) % res; // Bottom-right texel

    // Fetch the texel colors
    let c0 = textureLoad(texture, p0, 0).rgb;
    let c1 = textureLoad(texture, p1, 0).rgb;
    let c2 = textureLoad(texture, p2, 0).rgb;
    let c3 = textureLoad(texture, p3, 0).rgb;

    // Perform bilinear interpolation
    let cx0 = mix(c0, c1, ab_frac.x); // Interpolate horizontally top row
    let cx1 = mix(c2, c3, ab_frac.x); // Interpolate horizontally bottom row
    let texcolor = mix(cx0, cx1, ab_frac.y); // Interpolate vertically between the two rows

    return texcolor;
}


fn raytrace(r: ptr<function, Ray>, t: ptr<function, u32>) -> vec4f {
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    //const bgcolor = vec4f(0.0, 0.0, 0.0, 1.0);
    const max_depth = 10;
    var result = vec3f(0.0);
    var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), 0, vec3f(0.0), vec3f(0.0), vec2f(0.0), true, vec3f(1.0), vec3f(0.0));
    for (var i = 0; i < max_depth; i++) {
        if(intersect_scene(r, &hit)) {
            result += shade(r, &hit, t) * hit.factor;
        } else {
            result += bgcolor.rgb;
            break;
        }
        if(hit.has_hit) { break; }
    }
    return vec4f(pow(result, vec3f(1.0/uniforms_f.gamma)), bgcolor.a);
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
    let launch_idx = u32(fragcoord.y)*uniforms_ui.width + u32(fragcoord.x);
    var t = tea(launch_idx, uniforms_ui.frame);
    let jitter = vec2f(rnd(&t), rnd(&t))/f32(uniforms_ui.height);

    var uv = vec2f(coords.x*uniforms_f.aspect*0.5f, coords.y*0.5f);
    var r = get_camera_ray(uv + jitter, &t);
    var result = raytrace(&r, &t).rgb;

    // Progressive update of image
    let curr_sum = textureLoad(renderTexture, vec2u(fragcoord.xy), 0).rgb*f32(uniforms_ui.frame);
    let accum_color = (result + curr_sum)/f32(uniforms_ui.frame + 1u);
    var fsOut: FSOut;
    fsOut.frame = vec4f(pow(accum_color, vec3f(1.0/uniforms_f.gamma)), 1.0);
    fsOut.accum = vec4f(accum_color, 1.0);
    return fsOut;
}
