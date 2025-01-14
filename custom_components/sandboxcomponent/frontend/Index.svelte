<script lang="ts">
  import { onMount } from "svelte";
  import { debounce } from 'lodash';

  import "./style.css"; // Enable Tailwind CSS
  import IFrameContainer from "./shared/IFrameContainer.svelte";
  import type { SandboxMessage, UserInteraction } from "./types";

  // Gradio Utils
  import type { Gradio } from "@gradio/utils";
  import { StatusTracker } from "@gradio/statustracker";
  import type { LoadingStatus } from "@gradio/statustracker";
  import { Block } from "@gradio/atoms";

  // Gradio Props
  export let value: SandboxMessage | null;
  export let elem_id = "";
  export let elem_classes: string[] = [];
  // export let scale: number | null = null;
  export let min_width: number | undefined = undefined;
  export let loading_status: LoadingStatus;

  export let label: string;
  export let visible = true;

  export let gradio: Gradio<{
    change: never;
  }>;

  let sandboxUrl = value?.sandboxUrl;
  let iframeUserInteractions = value?.userInteractionRecords; // used to record user interactions in the iframe
  $: {
    if (value) {
      sandboxUrl = value.sandboxUrl;
      iframeUserInteractions = value.userInteractionRecords;
    }
  }

  const debouncedDispatchChange = debounce(() => {
    gradio.dispatch("change");
  }, 5 * 1000); // wait 5 seconds to avoid flooding the gradio server

  $: updateUserInteractions = (newInteraction: UserInteraction) => {
    if (!value) {
      return;
    }
    // console.log("updateUserInteractions", newInteraction);
    if (!value.enableTelemetry) {
      return;
    }
    value.userInteractionRecords = [
      ...value.userInteractionRecords,
      newInteraction,
    ];
    debouncedDispatchChange(); // Trigger Gradio update
  };

  let iframeKey = 0; // used to force re-render of iframe
  function refreshHTML() {
    iframeKey += 1;
  }

  export let width: number | null = null; // Initialize width to full window width
  export let height: number | null = 400; // Initial height in pixels
  let isResizing = false;
  let startY: number;
  let startX: number;
  let startHeight: number;
  let startWidth: number;
  let divElement: HTMLElement | null;

  function onClickResize(event: MouseEvent) {
    isResizing = true;
    startY = event.clientY;
    startX = event.clientX;
    startHeight = divElement?.clientHeight ?? 0;
    startWidth = divElement?.clientWidth ?? 0;
    // console.log("onClickResize", "startHeight", startHeight);
    // console.log("onClickResize", "startWidth", startWidth);
    window.addEventListener("mousemove", onResizeMouseMove);
  }

  function stopResizing() {
    isResizing = false;
    window.removeEventListener("mousemove", onResizeMouseMove);
  }

  function onResizeMouseMove(event: MouseEvent) {
    if (isResizing) {
      if (event.buttons === 0) {
        stopResizing();
      } else {
        height = Math.max(100, startHeight + (event.clientY - startY));
        width = Math.max(
          min_width ?? 100,
          startWidth + (event.clientX - startX)
        );
      }
    }
  }

  let isFullScreen = false;
  function toggleFullScreen() {
    isFullScreen = !isFullScreen;
    if (isFullScreen) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }

  let isDarkMode = false;
  function toggleDarkMode() {
    isDarkMode = !isDarkMode;
  }

  onMount(() => {
    document.addEventListener("fullscreenchange", () => {
      if (!document.fullscreenElement) {
        isFullScreen = false;
      }
    });
  });
</script>

<Block {visible} {elem_id} {elem_classes} container={false}>
  {#if loading_status}
    <StatusTracker
      autoscroll={gradio.autoscroll}
      i18n={gradio.i18n}
      {...loading_status}
      variant="center"
    />
  {/if}

  {#if sandboxUrl && sandboxUrl.length > 0}
    <div class:full-screen={isFullScreen}>
      <div
        class={`flex items-center justify-between border border-gray-300 rounded px-2 py-1 font-mono w-full ${isDarkMode ? "bg-black text-white" : "bg-white text-black"}`}
      >
        <div
          class={`p-2 rounded-t-md flex items-center mr-10 ${isDarkMode ? "bg-gray-900" : "bg-gray-100"}`}
        >
          <span
            class={`px-3 py-1 rounded-r-md flex-1 ${isDarkMode ? "text-white bg-gray-800" : "text-black bg-gray-200"}`}
          >
            {sandboxUrl}
          </span>
        </div>
        <div>
          <!-- Copy URL Button -->
          <button
            on:click={() => navigator.clipboard.writeText(sandboxUrl ?? '')}
            class="px-2 py-1 bg-green-500 text-white rounded cursor-pointer ml-2"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="1.5"
              class="size-5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25ZM6.75 12h.008v.008H6.75V12Zm0 3h.008v.008H6.75V15Zm0 3h.008v.008H6.75V18Z"
              />
            </svg>
          </button>
          <!-- Dark Mode Button -->
          <button
            on:click={toggleDarkMode}
            class="px-2 py-1 bg-blue-500 text-white rounded cursor-pointer ml-2"
          >
            {#if isDarkMode}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="1.5"
                stroke="currentColor"
                class="size-5"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z"
                />
              </svg>
            {:else}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="1.5"
                stroke="currentColor"
                class="size-5"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M12 3v1.5M12 19.5V21M4.22 4.22l1.06 1.06M17.72 17.72l1.06 1.06M3 12h1.5M19.5 12H21M4.22 19.78l1.06-1.06M17.72 6.28l1.06-1.06M12 6.75A5.25 5.25 0 1 0 17.25 12 5.25 5.25 0 0 0 12 6.75z"
                />
              </svg>
            {/if}
          </button>
          <!-- Full-Screen Button -->
          <button
            on:click={toggleFullScreen}
            class="px-2 py-1 bg-red-500 text-white rounded cursor-pointer"
          >
            {#if isFullScreen}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="1.5"
                stroke="currentColor"
                class="size-5"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25"
                />
              </svg>
            {:else}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="1.5"
                stroke="currentColor"
                class="size-5"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15"
                />
              </svg>
            {/if}
          </button>
          <!-- Open in new page Button -->
          <!-- <button on:click={() => window.open(sandboxUrl)} class="px-2 py-1 bg-blue-500 text-white rounded cursor-pointer ml-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
            </svg>
          </button> -->
          <!-- Refresh Button -->
          <button
            on:click={refreshHTML}
            class="px-2 py-1 bg-blue-500 text-white rounded cursor-pointer"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="1.5"
              stroke="currentColor"
              class="size-5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
              />
            </svg>
          </button>
        </div>
      </div>

      <div
        class:pending={loading_status?.status === "pending"}
        class="relative w-full flex flex-col"
        style={isFullScreen || !width || !height
          ? `height: 100%; width: 100%; max-width: 100%; max-height: 100%; min-height: 400px;`
          : `height: ${height}px; width: ${width}px; max-width: 100%; max-height: 100%; min-height: 400px;`}
        bind:this={divElement}
      >
        {#key iframeKey}
          <div
            class={`flex flex-col flex-1 p-10 ${isDarkMode ? "bg-slate-800" : "bg-slate-100"} backdrop-blur`}
          >
            <IFrameContainer
              min_height={loading_status &&
                loading_status?.status !== "complete"}
              value={sandboxUrl}
              {isDarkMode}
              {elem_classes}
              {visible}
              {updateUserInteractions}
            />
          </div>
        {/key}

        <div
          id="resizer"
          class="w-[15px] h-[15px] bg-gray-500 absolute right-0 bottom-0 cursor-[se-resize]"
          aria-hidden="true"
          on:mousedown|preventDefault={onClickResize}
        />
      </div>
    </div>
  {/if}
</Block>

<style>
  .full-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8); /* Modal overlay effect */
    z-index: 1000;
  }
</style>
