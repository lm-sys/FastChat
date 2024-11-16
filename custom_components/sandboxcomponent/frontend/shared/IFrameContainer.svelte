<script lang="ts">
  import { createEventDispatcher } from "svelte";
  export let elem_classes: string[] = [];
  export let value: string;
  export let visible = true;
  export let min_height = false;

  export let height: number;
  export let width: number;

  const dispatch = createEventDispatcher<{ change: undefined }>();

  let iframeElement;

  const onLoad = () => {
    try {
      const iframeDocument =
        iframeElement.contentDocument || iframeElement.contentWindow.document;
      if (height === 100) {
        iframeElement.style.height = `${iframeDocument.documentElement.scrollHeight}px`;
      } else {
        iframeElement.style.height = `${height}px`;
      }
    } catch (e) {
      console.error("Error accessing iframe content:", e);
    }
  };

  $: value, dispatch("change");

  $: if (iframeElement) {
    iframeElement.style.width = `${width}px`;
    iframeElement.style.height = `${height}px`;
  }
</script>

<div
  class="prose {elem_classes.join(' ')}"
  class:min={min_height}
  class:hide={!visible}
  class:height
  style={`position: relative; width: ${width}px; height: ${height}px;`}
>
  <iframe
    bind:this={iframeElement}
    title="iframe component"
    src={value}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen
    on:load={onLoad}
    width={width}
    height={height}
  ></iframe>
</div>

<style>
  .min {
    min-height: var(--size-24);
  }
  .hide {
    display: none;
  }
  .prose {
    overflow: auto;
  }
</style>
