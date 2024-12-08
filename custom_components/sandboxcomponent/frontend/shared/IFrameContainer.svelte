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
  class={`relative w-full h-full ${min_height ? "min" : ""} ${!visible ? "hidden" : ""} bg-white`}
>
  <iframe
    id="sandboxIframe"
    title="iframe component"
    src={value}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; display-capture"
    sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"
    allowtransparency
    allowfullscreen
    class="w-full h-full"
  />
</div>