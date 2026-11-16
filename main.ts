// make.fun ai-excited-01 entry driver.
// Copies the pre-rendered entry.png (produced by gen.py) from the repo root
// into the judge's output directory. Node 22, stdlib only.
import { copyFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";

const repoRoot = import.meta.dirname;
const src = join(repoRoot, "entry.png");
const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";

mkdirSync(outDir, { recursive: true });
const dest = join(outDir, "entry.png");
copyFileSync(src, dest);

console.log(`copied ${src} -> ${dest}`);
