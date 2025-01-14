import React, { useEffect, useRef, useCallback } from "react";
import { debounce } from 'lodash';

/**
 * A helper that posts a standardized message to the parent window.
 * Here, we include a timestamp in ISO format automatically.
 */
function postInteractionMessage(
  type: string,
  payload: Record<string, unknown> = {}
) {
  window.parent.postMessage(
    {
      type,
      time: new Date().toISOString(),
      ...payload,
    },
    "*"
  );
}

const App: React.FC = () => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const iframeDocRef = useRef<Document | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  // Get the iframeUrl from the URL parameter
  const searchParams = new URLSearchParams(window.location.search);
  const appParam = searchParams.get("app");
  const iframeUrl = appParam ? `/${appParam}/` : null;

  /**
   * Record size changes of the iframe element itself.
   * The Svelte code tried to read iframe?.clientWidth and clientHeight,
   * which can be done here with a ResizeObserver in React.
   */
  const recordIframeSize = useCallback(() => {
    if (iframeRef.current) {
      const width = iframeRef.current.clientWidth;
      const height = iframeRef.current.clientHeight;
      postInteractionMessage("resize", { width, height });
    }
  }, []);

  /**
   * Keydown event inside iframe
   */
  const handleIframeKeyDown = useCallback((e: KeyboardEvent) => {
    postInteractionMessage("keydown", { key: e.key });
  }, []);

  /**
   * Click event inside iframe
   */
  const handleIframeClick = useCallback((e: MouseEvent) => {
    postInteractionMessage("click", { x: e.clientX, y: e.clientY });
  }, []);

  /**
   * Scroll event inside iframe
   * debounce to reduce spam.
   */
  const handleIframeScroll = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    debounce((_e: Event) => {
      if (iframeDocRef.current) {
        const scrollTop = iframeDocRef.current.documentElement.scrollTop;
        const scrollLeft = iframeDocRef.current.documentElement.scrollLeft;
        postInteractionMessage("scroll", { scrollTop, scrollLeft });
      }
    }, 50),
    []
  );

  /**
   * When the iframe loads, attempt to:
   * 1) post "load" message,
   * 2) attach a ResizeObserver,
   * 3) attach event listeners for keydown, click, scroll, etc.
   */
  const handleIframeLoad = useCallback(() => {
    try {
      // Post message that the iframe has loaded
      postInteractionMessage("load");

      // Record initial size
      recordIframeSize();

      // Start observing the iframe size changes
      if (!resizeObserverRef.current) {
        resizeObserverRef.current = new ResizeObserver(
          debounce(recordIframeSize, 1000)
        );
      }
      if (iframeRef.current) {
        resizeObserverRef.current.observe(iframeRef.current);
      }

      // Detach existing listeners if any
      if (iframeDocRef.current) {
        iframeDocRef.current.removeEventListener("keydown", handleIframeKeyDown);
        iframeDocRef.current.removeEventListener("click", handleIframeClick);
        iframeDocRef.current.removeEventListener("scroll", handleIframeScroll);
        iframeDocRef.current = null;
      }

      // Attach new listeners if we can access the iframe document
      const newDoc = iframeRef.current?.contentWindow?.document;
      if (newDoc) {
        iframeDocRef.current = newDoc;

        newDoc.addEventListener("keydown", handleIframeKeyDown);
        newDoc.addEventListener("click", handleIframeClick);
        newDoc.addEventListener("scroll", handleIframeScroll, { passive: true });
      } else {
        console.warn(
          "Iframe is loaded but no document found (possible cross-origin issue)."
        );
      }
    } catch (err) {
      console.error("Failed to attach iframe event listeners:", err);
      postInteractionMessage("captureError", { error: String(err) });
    }
  }, [
    recordIframeSize,
    handleIframeKeyDown,
    handleIframeClick,
    handleIframeScroll
  ]);

  /**
   * On unmount, clean up event listeners and the resize observer.
   */
  useEffect(() => {
    return () => {
      // Clean up the resize observer
      if (resizeObserverRef.current && iframeRef.current) {
        resizeObserverRef.current.unobserve(iframeRef.current);
      }
      resizeObserverRef.current = null;

      // Clean up any iframe doc listeners
      if (iframeDocRef.current) {
        iframeDocRef.current.removeEventListener("keydown", handleIframeKeyDown);
        iframeDocRef.current.removeEventListener("click", handleIframeClick);
        iframeDocRef.current.removeEventListener("scroll", handleIframeScroll);
        iframeDocRef.current = null;
      }
    };
  }, [handleIframeKeyDown, handleIframeClick, handleIframeScroll]);

  return iframeUrl ? (
    <iframe
      ref={iframeRef}
      src={iframeUrl} // Use the iframeUrl here
      title="app iframe"
      style={{
        width: "100%",
        height: "100%",
        border: "none",
        margin: 0,
        padding: 0,
      }}
      onLoad={handleIframeLoad}
    />
  ) : (
    <div>No app selected</div>
  );
};

export default App;