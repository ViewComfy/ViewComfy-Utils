import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";


// Register extension for ViewComfy-Utils nodes
app.registerExtension({
    name: "ViewComfy.Utils.ShowAnything",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const node_name = nodeData.name;

        // Handle showAnything_ViewComfy node
        if (node_name === 'showAnything_ViewComfy') {
            function populate(text, name = 'text') {
                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === name);
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }
                
                for (const list of text) {
                    const w = ComfyWidgets["STRING"](this, "text", ["STRING", {multiline: true}], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = list;
                }
                
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            const onExecuted = nodeType.prototype.onExecuted;
            // When the node is executed we will be sent the input text, display this in the widget
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                populate.call(this, message.text, 'text');
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values, 'text');
                }
            };
        }

        // Handle anythingInversedSwitch_ViewComfy node - dynamic outputs
        if (node_name === 'anythingInversedSwitch_ViewComfy') {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                onNodeCreated?.apply(this, arguments);
                // Remove extra outputs on creation, keeping only the first one
                this.outputs = this.outputs.filter((cate, index) => index <= 0);
            };

            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (!link_info) return;
                
                // Only handle output connections (type == 2)
                if (type == 2) {
                    let outputs = this.outputs;
                    let is_output_all_connected = outputs.every(cate => cate.links?.length > 0);
                    
                    if (is_output_all_connected) {
                        // All outputs are connected, add a new output
                        if (outputs.length >= 20) {
                            console.warn('ViewComfy-Utils: The maximum number of outputs is 20');
                            return;
                        }
                        let output_label = 'out' + outputs.length;
                        this.addOutput(output_label, '*');
                    } else if (!connected) {
                        // An output was disconnected
                        let slot_index = link_info.origin_slot;
                        if (slot_index == this.outputs.length - 2 && outputs[slot_index].links?.length == 0) {
                            // If we disconnected the second-to-last output and it has no links, remove the last output
                            this.removeOutput(slot_index + 1);
                        }
                    }
                }
            };
        }
    },
});

