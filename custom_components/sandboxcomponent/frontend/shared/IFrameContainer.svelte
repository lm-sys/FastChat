<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export const elem_classes: string[] = [];
  export let value: string;
  export let visible = true;
  export let min_height = false;
  export let isDarkMode = false;

  const dispatch = createEventDispatcher<{ change: undefined }>();

  $: value, dispatch("change");
</script>

<div
  class={`relative flex flex-col flex-1 w-full h-full ${min_height ? "min" : ""} ${!visible ? "hidden" : ""} ${isDarkMode ? "bg-black text-white" : "bg-white text-black"}`}
>
  <iframe
    id="sandboxIframe"
    title="iframe component"
    src={value}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; display-capture"
    sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts"
    allowtransparency
    allowfullscreen
    class="w-full h-full flex-1"
  />
</div>
