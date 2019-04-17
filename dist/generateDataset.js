"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const iq = require("inquirer");
const shelljs = require("shelljs");
function promptForOptions() {
    return __awaiter(this, void 0, void 0, function* () {
        return yield iq.prompt([
            {
                type: "input",
                name: "name",
                message: "Name of the dataset",
                default: "default"
            },
            {
                type: "number",
                name: "clientCount",
                message: "Number of clients",
                default: 500
            },
            {
                type: "number",
                name: "minSamples",
                message: "Minimum number of samples",
                default: 200
            },
            {
                type: "number",
                name: "maxSamples",
                message: "Maximum number of samples",
                default: 1000
            },
            {
                type: "number",
                name: "minClasses",
                message: "Minimum number of classes",
                default: 3
            },
            {
                type: "number",
                name: "maxClasses",
                message: "Maximum number of classes",
                default: 10
            },
            {
                type: "confirm",
                name: "repeat",
                message: "Allow repeated data among clients?",
                default: true
            }
        ]);
    });
}
function generateDataset() {
    return __awaiter(this, void 0, void 0, function* () {
        const options = yield promptForOptions();
        shelljs.env.TF_CPP_MIN_LOG_LEVEL = "3";
        const { stderr, stdout, code } = shelljs.exec([
            "python python/setup.py",
            "-n", options.name,
            "-t", options.clientCount,
            "-s", options.minSamples,
            "-u", options.maxSamples,
            "-v", options.minClasses,
            "-w", options.maxClasses,
            "-r", options.repeat
        ].join(" "));
        if (code !== 0) {
            console.error("[Simulator] Setup failed");
            process.exit(1);
        }
    });
}
generateDataset();
//# sourceMappingURL=generateDataset.js.map