import tailwindcss from "@tailwindcss/vite";
export default {
  plugins: [
    tailwindcss(),
  ],
  svelte: {
    preprocess: [],
  },
  build: {
    target: "modules",
  },
};