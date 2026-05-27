const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  webpack: {
    plugins: {
      add: [
        new CopyPlugin({
          patterns: [
            { from: "node_modules/onnxruntime-web/dist/*.wasm", to: "static/js/[name][ext]" },
            { from: './public/model/*.onnx', to: '[name][ext]' }
          ],
        }),
      ],
    },
    configure: (config) => {
      config.resolve.fallback = {
        fs: false,
        path: false,
        crypto: false,
      };
      return config;
    },
  },
};
