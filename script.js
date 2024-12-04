"use strict";
window.onload = function () { main(); }
async function main() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    const response = await fetch("./shaders.wgsl");
    const wgsl = device.createShaderModule({
        code: await response.text()
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: wgsl,
            entryPoint: "main_vs",
        },
        fragment: {
            module: wgsl,
            entryPoint: "main_fs",
            targets: [{ format: canvasFormat },
                      { format: "rgba32float" }]
        },
        primitive: {
            topology: "triangle-strip",
        },
    });

    // Create buffers for object data
    const obj_filename = '../data/objects/CornellBox.obj';
    const drawingInfo = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices

    let buffers = new Object();
    buffers = build_bsp_tree(drawingInfo, device, buffers);

    const attributesBuffer = device.createBuffer({
        size: drawingInfo.attribs.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(attributesBuffer, 0, drawingInfo.attribs);

    const indicesBuffer = device.createBuffer({
        size: drawingInfo.indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(indicesBuffer, 0, drawingInfo.indices);

    const materialsBuffer = device.createBuffer({
        size: drawingInfo.materials.length * (4 * 4 * 4), // 4 vec4
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    const parsedMaterials = drawingInfo.materials.map((x) => {
        var illum = 0.0;
        if (x.illum == 1) {
            illum = 1.0;
        }
        let obj = [
            x.color.r, x.color.g, x.color.b, x.color.a,
            x.emission.r, x.emission.g, x.emission.b, x.emission.a,
            x.specular.r, x.specular.g, x.specular.b, x.specular.a,
            illum, 0.0, 0.0, 0.0 // to pass it like a float
        ];
        return obj;
    });
    device.queue.writeBuffer(materialsBuffer, 0, flatten(parsedMaterials));

    const light_indicesBuffer = device.createBuffer({
        size: drawingInfo.light_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(light_indicesBuffer, 0, drawingInfo.light_indices);


    // Create one texture as a render target and one to load the previous result from
    let textures = new Object();
    textures.width = canvas.width;
    textures.height = canvas.height;
    textures.renderSrc = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        format: 'rgba32float',
    });
    textures.renderDst = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        format: 'rgba32float',
    });


    // Sliders
    var l_cam = 2.0; var zd_cam = 5.0; var cam_const = 1.0;
    var l_proj = 2.0; var zd_proj = 5.0; var proj_const = 1.0;
    document.getElementById("l-cam").oninput = function() {updateSliders()};
    document.getElementById("zd-cam").oninput = function() {updateSliders()};
    document.getElementById("cam-const").oninput = function() {updateSliders()};
    document.getElementById("l-proj").oninput = function() {updateSliders()};
    document.getElementById("zd-proj").oninput = function() {updateSliders()};
    document.getElementById("proj-const").oninput = function() {updateSliders()};

    function updateSliders() {
        // Gets the input value
        l_cam = document.getElementById("l-cam").value
        uniforms_f[2] = l_cam;
        // Displays this value to the html page
        document.getElementById('l-cam-output').innerHTML = parseFloat(l_cam).toFixed(1);

        zd_cam = document.getElementById("zd-cam").value
        uniforms_f[3] = zd_cam;
        document.getElementById('zd-cam-output').innerHTML = parseFloat(zd_cam).toFixed(1);

        cam_const = document.getElementById("cam-const").value
        uniforms_f[4] = cam_const;
        document.getElementById('cam-const-output').innerHTML = parseFloat(cam_const).toFixed(1);

        l_proj = document.getElementById("l-proj").value
        uniforms_f[5] = l_proj;
        document.getElementById('l-proj-output').innerHTML = parseFloat(l_proj).toFixed(1);

        zd_proj = document.getElementById("zd-proj").value
        uniforms_f[6] = zd_proj;
        document.getElementById('zd-proj-output').innerHTML = parseFloat(zd_proj).toFixed(1);

        proj_const = document.getElementById("proj-const").value
        uniforms_f[7] = proj_const;
        document.getElementById('proj-const-output').innerHTML = parseFloat(proj_const).toFixed(1);

        device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);

        // Reset frame counter
        frame = 0;
    };

    // Uniform float buffer
    var aspect = canvas.width/canvas.height;
    var gamma = 2.25;
    var uniforms_f = new Float32Array([aspect, gamma,
                                       l_cam, zd_cam, cam_const,
                                       l_proj, zd_proj, proj_const]);
    const uniformBuffer_f = device.createBuffer({
        size: uniforms_f.byteLength, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);


    // Menu selection    
    var menu_1 = document.getElementById("menu1");
    var shader_1 = menu_1.selectedIndex;
    menu_1.addEventListener("change", () => { uniforms_ui[0] = menu_1.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        frame = 0;
        requestAnimationFrame(animate); });

    var menu_2 = document.getElementById("menu2");
    var shader_2 = menu_2.selectedIndex;
    menu_2.addEventListener("change", () => { uniforms_ui[1] = menu_2.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        frame = 0;
        requestAnimationFrame(animate); });

    // Checkboxes
    var prog_update = true;
    document.getElementById("prog_update").onclick = function() {
        if (prog_update) { console.log(uniforms_ui[4]) };
        prog_update = !prog_update;
        animate();
    };

    var dir_light = true;
    document.getElementById("dir_light").onclick = function() {
        dir_light = !dir_light;
        uniforms_ui[7] = dir_light;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        frame = 0;
        animate();
    };

    var proj_light = true;
    document.getElementById("proj_light").onclick = function() {
        proj_light = !proj_light;
        uniforms_ui[8] = proj_light;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        frame = 0;
        animate();
    };

    var indir_light = true;
    document.getElementById("indir_light").onclick = function() {
        indir_light = !indir_light;
        uniforms_ui[9] = indir_light;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        frame = 0;
        animate();
    };


    // Create texture for the projector
    const texture = await load_texture(device, './data/img.jpg');
    const texture_width = texture.width;
    const texture_height = texture.height;
    const textureView = texture.createView();

    // Write values to uniform buffers
    var width = canvas.width; var height = canvas.height; var frame = 0;
    var uniforms_ui = new Uint32Array([shader_1, shader_2,
                                       width, height, frame,
                                       texture_width, texture_height,
                                       dir_light, proj_light, indir_light]);
    const uniformBuffer_ui = device.createBuffer({
        size: uniforms_ui.byteLength, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);


    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer_f }},
            { binding: 1, resource: { buffer: uniformBuffer_ui }},
            // { binding: 2, resource: { buffer: attributesBuffer }},
            // { binding: 3, resource: { buffer: indicesBuffer }},
            // { binding: 4, resource: { buffer: materialsBuffer }},
            // { binding: 5, resource: { buffer: light_indicesBuffer }},
            // { binding: 6, resource: { buffer: buffers.aabb }},
            // { binding: 7, resource: { buffer: buffers.treeIds }},
            // { binding: 8, resource: { buffer: buffers.bspTree }},
            // { binding: 9, resource: { buffer: buffers.bspPlanes }},
            { binding: 10, resource: textures.renderDst.createView() },
            { binding: 11, resource: textureView }
        ],
    });

    function animate() {
        if (prog_update) {
            uniforms_ui[4] = frame; frame += 1;
            device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
            requestAnimationFrame(animate);
        }

        render(device, context, pipeline, bindGroup, textures);
    }
    animate();
}

function render(device, context, pipeline, bindGroup, textures) {
    // Create a render pass in a command buffer and submit it
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [
            { view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" },
            { view: textures.renderSrc.createView(), loadOp: "load", storeOp: "store" }
        ]
    });
    
    // Insert render pass commands here
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4);
    pass.end();

    encoder.copyTextureToTexture({ texture: textures.renderSrc }, { texture: textures.renderDst },
        [textures.width, textures.height]);
    
    // Finish the command buffer and immediately submit it.
    device.queue.submit([encoder.finish()]);
}

async function load_texture(device, filename) {
    const response = await fetch(filename);
    const blob = await response.blob();
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
    const texture = device.createTexture({
        size: [img.width, img.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.copyExternalImageToTexture(
        { source: img, flipY: true },
        { texture: texture },
        { width: img.width, height: img.height },
    );
    return texture;
}
