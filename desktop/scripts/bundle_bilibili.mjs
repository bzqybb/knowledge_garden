import { build } from "esbuild";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  throw new Error("usage: node bundle_bilibili.mjs <input> <output>");
}

await build({
  entryPoints: [input],
  outfile: output,
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  banner: {
    js: 'const __import_meta_url = require("node:url").pathToFileURL(__filename).href;',
  },
  define: {
    "import.meta.url": "__import_meta_url",
  },
  logLevel: "warning",
});
