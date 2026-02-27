const nextJest = require("next/jest");
const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jest-environment-jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    // d3 v7 packages ship ESM-only source (main → src/index.js).
    // Jest runs in CommonJS mode and can't parse ESM in node_modules without
    // explicit transform config.  The UMD dist builds (CJS-compatible) are a
    // simpler fix: map every d3-* import to its prebuilt UMD bundle.
    "^(d3-.+)$": "<rootDir>/node_modules/$1/dist/$1.js",
  },
  testRegex: "src/__tests__/.*\\.test\\.(ts|tsx)$",
};

module.exports = createJestConfig(config);
