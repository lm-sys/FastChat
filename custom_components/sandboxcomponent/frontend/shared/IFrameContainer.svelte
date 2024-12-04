<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export const elem_classes: string[] = [];
  export let value: string;
  export let visible = true;
  export let min_height = false;

  const dispatch = createEventDispatcher<{ change: undefined }>();

  $: value, dispatch("change");
</script>

<div
  class={`relative w-full h-full ${min_height ? "min" : ""} ${!visible ? "hidden" : ""}`}
>
  <iframe
    id="sandboxIframe"
    title="iframe component"
    src={value}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; display-capture"
    allowfullscreen
    class="w-full h-full"
  />
</div>
