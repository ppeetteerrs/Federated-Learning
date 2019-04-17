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
const bb = require("bluebird");
const child_process_1 = require("child_process");
const fs_1 = require("fs");
const path_1 = require("path");
const shelljs = require("shelljs");
const yargs = require("yargs");
const asyncShell_1 = require("./asyncShell");
const CLIOptions = yargs
    .options({
    name: {
        alias: "n",
        default: "default",
        describe: "Name of the run",
        type: "string"
    },
    dataset: {
        alias: "d",
        default: "default",
        describe: "Name of the dataset",
        type: "string"
    },
    gpu: {
        alias: "g",
        default: 0,
        describe: "ID of GPU to use",
        type: "number"
    },
    batchSize: {
        alias: "b",
        default: 64,
        describe: "Batch size on the client",
        type: "number"
    },
    epochs: {
        alias: "e",
        default: 1,
        describe: "Number of epochs on the client",
        type: "number"
    },
    clientsPerRound: {
        alias: "c",
        default: 15,
        describe: "Number of clients to choose per round",
        type: "number"
    },
    concurrency: {
        alias: "cc",
        default: 3,
        describe: "Number of clients to run simultaneously",
        type: "number"
    },
    serverGPUFraction: {
        alias: "gs",
        default: 0.2,
        describe: "GPU fraction for server",
        type: "number"
    },
    clientGPUFraction: {
        alias: "gc",
        default: 0.15,
        describe: "GPU fraction for each client",
        type: "number"
    },
    iterations: {
        alias: "i",
        default: 2000,
        describe: "Total number of iterations",
        type: "number"
    }
})
    .help()
    .argv;
class Simulator {
    constructor() {
        this.options = CLIOptions;
        this.setup = () => __awaiter(this, void 0, void 0, function* () {
            // Sanity Checks
            // Run Setup Script
            shelljs.env.TF_CPP_MIN_LOG_LEVEL = "3";
            shelljs.env.CUDA_VISIBLE_DEVICES = this.options.gpu.toString();
            this.serverProcess = child_process_1.spawn("python", [
                "python/server.py",
                "-c",
                this.options.clientsPerRound.toString(),
                "-t",
                this.getClientCount().toString(),
                "-d",
                this.options.dataset,
                "-i",
                this.options.iterations.toString(),
                "-n",
                this.options.name,
                "-f",
                this.options.serverGPUFraction.toFixed(10)
            ]);
            this.serverProcess.stderr.pipe(process.stderr);
            this.serverProcess.stdout.on("data", this.handleServerMessage);
            this.serverProcess.on("exit", () => {
                console.log(`[Simulator] Child Process Exited`);
            });
            this.startServer();
            return this;
        });
        this.handleServerMessage = (data) => {
            const message = JSON.parse(data.toString());
            switch (message.type) {
                case "train":
                    bb.map(message.clients, clientId => {
                        return asyncShell_1.execAsync(["python python/client.py",
                            "-n", this.options.name,
                            "-w", message.weights_file_path,
                            "-i", clientId,
                            "-s", message.step,
                            "-f", this.options.serverGPUFraction.toFixed(10),
                            "-d", this.options.dataset].join(" "), {
                            async: true,
                            silent: true,
                        });
                    }, {
                        concurrency: this.options.concurrency,
                    }).then(results => {
                        // Filter Results
                        try {
                            const successfulClients = results
                                .filter(value => value.code === 0)
                                .map(value => JSON.parse(value.stdout.split("\n").slice(-1)[0]).id);
                            // console.log(
                            //   `Clients ${successfulClients} (${successfulClients.length} / ${
                            //     message.clients.length
                            //   }) Trained `
                            // );
                            this.sendWeights(successfulClients);
                        }
                        catch (e) {
                            console.log(results);
                        }
                    });
                    break;
                case "update":
                    console.log(`[Simulator] Step ${message.step}: ${JSON.stringify(message.message, null, 4)}`);
                    break;
                case "log":
                default:
                    console.log(`[Simulator] Log From Server: ${JSON.stringify(message.message, null, 4)}`);
                    break;
            }
        };
    }
    getClientCount() {
        const datasetPath = path_1.join(process.cwd(), "datasets", this.options.dataset);
        const fileNames = fs_1.readdirSync(datasetPath).filter(name => name.endsWith(".h5") && name.startsWith("data_client"));
        return fileNames.length;
    }
    startServer() {
        this.serverProcess.stdin.write("start\n");
        console.log("[Simulator] Server started");
    }
    sendWeights(ids) {
        this.serverProcess.stdin.write(JSON.stringify({ ids }) + "\n");
    }
}
new Simulator().setup();
//# sourceMappingURL=simulator.js.map