import * as bb from "bluebird";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import { readdirSync } from "fs";
import { join } from "path";
import * as shelljs from "shelljs";
import * as yargs from "yargs";
import { execAsync } from "./asyncShell";

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

interface MessageFormat {
  type: "train" | "log" | "update";
  message?: string;
  clients?: number[];
  weights_file_path?: string;
  step?: number;
}

class Simulator {
  public options = CLIOptions;
  public serverProcess: ChildProcessWithoutNullStreams;

  public setup = async (): Promise<Simulator> => {
    // Sanity Checks
    // Run Setup Script

    shelljs.env.TF_CPP_MIN_LOG_LEVEL = "3";
    shelljs.env.CUDA_VISIBLE_DEVICES = this.options.gpu.toString();

    this.serverProcess = spawn("python", [
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
  };

  private getClientCount() {
    const datasetPath = join(process.cwd(), "datasets", this.options.dataset);
    const fileNames = readdirSync(datasetPath).filter(name => name.endsWith(".h5") && name.startsWith("data_client"));
    return fileNames.length;
  }

  private startServer() {
    this.serverProcess.stdin.write("start\n");
    console.log("[Simulator] Server started");
  }

  private sendWeights(ids: number[]) {
    this.serverProcess.stdin.write(JSON.stringify({ ids }) + "\n");
  }

  private handleServerMessage = (data: Buffer) => {
    const message: MessageFormat = JSON.parse(data.toString());

    switch (message.type) {
      case "train":
        bb.map<number, { stdout: string; stderr: string; code: number }>(
          message.clients,
          clientId => {
            return execAsync(
              ["python python/client.py",
                "-n", this.options.name,
                "-w", message.weights_file_path,
                "-i", clientId,
                "-s", message.step,
                "-f", this.options.clientGPUFraction.toFixed(10),
                "-d", this.options.dataset].join(" "),
              {
                async: true,
                silent: true,
              }
            );
          },
          {
            concurrency: this.options.concurrency,
          }
        ).then(results => {
          // Filter Results
          try {
            const successfulClients: number[] = results
              .filter(value => value.code === 0)
              .map(
                value => JSON.parse(value.stdout.split("\n").slice(-1)[0]).id
              );
            // console.log(
            //   `Clients ${successfulClients} (${successfulClients.length} / ${
            //     message.clients.length
            //   }) Trained `
            // );
            this.sendWeights(successfulClients);
          } catch (e) {
            console.log(results);
          }
        });
        break;
      case "update":
        console.log(
          `[Simulator] Step ${message.step}: ${JSON.stringify(
            message.message,
            null,
            4
          )}`
        );
        break;
      case "log":
      default:
        console.log(
          `[Simulator] Log From Server: ${JSON.stringify(
            message.message,
            null,
            4
          )}`
        );
        break;
    }
  };
}

new Simulator().setup();
