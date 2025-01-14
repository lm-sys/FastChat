interface BaseInteraction {
    type: string;
    time: string;
}

interface LoadInteraction extends BaseInteraction {
    type: "load";
}

interface KeydownInteraction extends BaseInteraction {
    type: "keydown";
    key: string;
}

interface ClickInteraction extends BaseInteraction {
    type: "click";
    x: number;
    y: number;
}

interface ScrollInteraction extends BaseInteraction {
    type: "scroll";
    scrollTop: number;
    scrollLeft: number;
}

interface ResizeInteraction extends BaseInteraction {
    type: "resize";
    width: number;
    height: number;
}

interface CaptureErrorInteraction extends BaseInteraction {
    type: "captureError";
    error: Error;
}

type UserInteraction =
    | LoadInteraction
    | KeydownInteraction
    | ClickInteraction
    | ScrollInteraction
    | ResizeInteraction
    | CaptureErrorInteraction;

type SandboxData = {
    sandboxUrl: string; // URL of the sandbox
    enableTelemetry: boolean; // Whether to collect user interactions
    userInteractionRecords: UserInteraction[]; // User Interaction history
}

export type { UserInteraction, SandboxData as SandboxMessage };