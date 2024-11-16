<script lang="ts">
  import type { Gradio } from "@gradio/utils";
  import IFrameContainer from "./shared/IFrameContainer.svelte";
  import { StatusTracker } from "@gradio/statustracker";
  import type { LoadingStatus } from "@gradio/statustracker";
  import { Block } from "@gradio/atoms";

  export let label: string;
  export let elem_id = "";
  export let elem_classes: string[] = [];
  export let visible = true;
  export let value: string  = "";
  export let height: number = 300; // Initial height in pixels
  export let width: number = 400; // Initial width in pixels
  export let loading_status: LoadingStatus;
  export let gradio: Gradio<{
    change: never;
  }>;

  $: label, gradio.dispatch("change");

  let [sandboxUrl, sandboxCode] = JSON.parse(value);
  console.log("DEBUG input", value, '1', sandboxUrl, sandboxCode);

  // Add a reactive statement for value
  $: {
    [sandboxUrl, sandboxCode] = JSON.parse(value);
    console.log("DEBUG input", value, '1', sandboxUrl, sandboxCode);
  }

  function refreshHTML() {
    // Logic to refresh the HTML component
    gradio.dispatch("change");
  }

  let isResizing = false;
  let startX: number;
  let startY: number;
  let startWidth: number;
  let startHeight: number;

  function onMouseDown(event: MouseEvent) {
    isResizing = true;
    startX = event.clientX;
    startY = event.clientY;
    startWidth = width;
    startHeight = height;
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }

  function onMouseMove(event: MouseEvent) {
    if (isResizing) {
      width = Math.max(100, startWidth + (event.clientX - startX));
      height = Math.max(100, startHeight + (event.clientY - startY));
    }
  }

  function onMouseUp() {
    isResizing = false;
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
  }

  let activeTab = 'html'; // Track the active tab

  function switchTab(tab: string) {
    activeTab = tab;
  }
</script>

<Block {visible} {elem_id} {elem_classes} container={false}>
  
<div
  style={`display: flex; align-items: center; background-color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; font-family: monospace; width: ${width}px;`}
>
    <span style="color: white;">URL: {sandboxUrl}</span>
    <button on:click={refreshHTML} style="margin-left: 10px; padding: 5px; border: none; background-color: #007bff; color: white; border-radius: 4px; cursor: pointer;">
      Refresh
    </button>
  </div>


  <StatusTracker
    autoscroll={gradio.autoscroll}
    i18n={gradio.i18n}
    {...loading_status}
    variant="center"
  />

    <!-- Tab buttons -->
  <!-- <div style={`display: flex; margin-top: 10px; width: ${width}px;`}>
    <button on:click={() => switchTab('html')} style="flex: 1; padding: 10px; border: none; background-color: {activeTab === 'html' ? '#007bff' : '#ccc'}; color: white; cursor: pointer;">
      HTML
    </button>
    <button on:click={() => switchTab('code')} style="flex: 1; padding: 10px; border: none; background-color: {activeTab === 'code' ? '#007bff' : '#ccc'}; color: white; cursor: pointer;">
      Source Code
    </button>
  </div> -->

  <!-- Tab content -->
  <div class:pending={loading_status?.status === "pending"} style={`position: relative; width: ${width}px; height: ${height}px;`}>
    <!-- {#if activeTab === 'html'} -->
      <IFrameContainer
        min_height={loading_status && loading_status?.status !== "complete"}
        value={sandboxUrl}
        {elem_classes}
        {visible}
        {width}
        {height}
        on:change={() => gradio.dispatch("change")}
      />
    <!-- {:else if activeTab === 'code'}
      <pre style="white-space: pre-wrap; word-wrap: break-word; background-color: black; color: white; padding: 10px; border-radius: 4px; overflow: auto; width: 100%; height: 100%;">
        {sandboxCode}
      </pre>
    {/if} -->
  
    <div
      on:mousedown|preventDefault={onMouseDown}
      style="
        width: 15px;
        height: 15px;
        background: #ccc;
        position: absolute;
        right: 0;
        bottom: 0;
        cursor: se-resize;"
    />
  </div>
</Block>
