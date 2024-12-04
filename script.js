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


    // Uniform float buffer
    var aspect = canvas.width/canvas.height;
    var cam_const = 1;
    var gamma = 2.25;
    var z_d = 5.0;
    var l = 0.1;

    var uniforms_f = new Float32Array([aspect, cam_const, gamma, z_d, l]);
    const uniformBuffer_f = device.createBuffer({
        size: uniforms_f.byteLength, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);


    // Keyboard inputs
    document.onkeydown = (event) => {
        switch(event.key) {
            case "ArrowRight":
                uniforms_f[0] += 0.1;
                device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
                requestAnimationFrame(animate);
                break
            case "ArrowLeft":
                uniforms_f[0] -= 0.1;
                device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
                requestAnimationFrame(animate);
                break
            case "ArrowUp":
                uniforms_f[1] += 0.1;
                device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
                requestAnimationFrame(animate);
                break
            case "ArrowDown":
                uniforms_f[1] -= 0.1;
                device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
                requestAnimationFrame(animate);
                break
        }
    };
    
    // Menu selection    
    var glassMenu = document.getElementById("glassMenu");
    var glass_shader = glassMenu.selectedIndex;
    glassMenu.addEventListener("change", () => { uniforms_ui[0] = glassMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate); });

    var matteMenu = document.getElementById("matteMenu");
    var matte_shader = matteMenu.selectedIndex;
    matteMenu.addEventListener("change", () => { uniforms_ui[1] = matteMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate); });

    var textureMenu = document.getElementById("useTexture");
    const use_texture = textureMenu.selectedIndex;
    textureMenu.addEventListener("click", () => { uniforms_ui[2] = textureMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate);
    });

    var addressMenu = document.getElementById("addressmode");
    const use_repeat = addressMenu.selectedIndex;
    addressMenu.addEventListener("click", () => { uniforms_ui[3] = addressMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate);
    });

    var filterMenu = document.getElementById("filtermode");
    const use_linear = filterMenu.selectedIndex;
    filterMenu.addEventListener("click", () => { uniforms_ui[4] = filterMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate);
    });

    var prog_update = true;
    document.getElementById("prog_update").onclick = function() {
        if (prog_update) { console.log(uniforms_ui[7]) };
        prog_update = !prog_update;
        animate();
    };

    // Write values to uniform buffers
    var width = canvas.width; var height = canvas.height; var frame = 0;
    var uniforms_ui = new Uint32Array([glass_shader, matte_shader,
                                       use_texture, use_repeat, use_linear,
                                       width, height, frame]);
    const uniformBuffer_ui = device.createBuffer({
        size: uniforms_ui.byteLength, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);

    // create texture for the projector

    const texture = await load_texture(device, './data/check.png');
    const textureView = texture.createView();

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
            uniforms_ui[7] = frame; frame += 1;
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
