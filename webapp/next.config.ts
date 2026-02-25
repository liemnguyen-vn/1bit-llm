import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "export",
  basePath: isProd ? "/1bit-llm" : "",
  assetPrefix: isProd ? "/1bit-llm/" : "",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
