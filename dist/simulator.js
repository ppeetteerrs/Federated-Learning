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
const inquirer = require("inquirer");
const shelljs = require("shelljs");
const asyncShell_1 = require("./asyncShell");
class Simulator {
    constructor() {
        this.setup = () => __awaiter(this, void 0, void 0, function* () {
            // Sanity Checks
            // Prompt for Settings
            const { name, localBatchSize, localEpochs, clientsPerRound, clientCount, iterations, } = yield inquirer.prompt([
                {
                    type: 'input',
                    name: 'name',
                    message: 'What is the name of this simulation?',
                    default: 'default',
                },
                {
                    type: 'input',
                    name: 'localBatchSize',
                    message: 'What is the local batch size?',
                    default: 64,
                },
                {
                    type: 'input',
                    name: 'localEpochs',
                    message: 'What is the local epochs?',
                    default: 2,
                },
                {
                    type: 'input',
                    name: 'clientsPerRound',
                    message: 'How many clients per round?',
                    default: 10,
                },
                {
                    type: 'input',
                    name: 'clientCount',
                    message: 'How many clients in total?',
                    default: 300,
                },
                {
                    type: 'input',
                    name: 'iterations',
                    message: 'How many iterations in total?',
                    default: 10000,
                },
            ]);
            this.name = name;
            this.localBatchSize = localBatchSize;
            this.localEpochs = localEpochs;
            this.clientsPerRound = clientsPerRound;
            this.clientCount = clientCount;
            this.iterations = iterations;
            // Run Setup Script
            const { stderr, stdout, code } = shelljs.exec(`python python/setup.py -n ${this.name} -b ${this.localBatchSize} -e ${this.localEpochs} -t ${this.clientCount}`);
            if (code !== 0) {
                console.error('[Simulator] Setup failed');
                process.exit(1);
            }
            console.log('[Simulator] Finished setting up');
            this.serverProcess = child_process_1.spawn('python', [
                'python/server.py',
                '-c',
                this.clientsPerRound.toString(),
                '-t',
                this.clientCount.toString(),
                '-i',
                this.iterations.toString(),
                '-n',
                this.name,
            ]);
            this.serverProcess.stderr.pipe(process.stderr);
            this.serverProcess.stdout.on('data', this.handleServerMessage);
            this.serverProcess.on('exit', () => {
                console.log(`[Simulator] Child Process Exited`);
            });
            this.startServer();
            return this;
        });
        this.handleServerMessage = (data) => {
            const message = JSON.parse(data.toString());
            switch (message.type) {
                case 'train':
                    bb.map(message.clients, clientId => {
                        return asyncShell_1.execAsync(`python python/client.py -n ${this.name} -w ${message.weights_file_path} -i ${clientId} -s ${message.step}`, {
                            async: true,
                            silent: true,
                        });
                    }, {
                        concurrency: 1,
                    }).then(results => {
                        // Filter Results
                        try {
                            const successfulClients = results
                                .filter(value => value.code === 0)
                                .map(value => JSON.parse(value.stdout.split('\n').slice(-1)[0]).id);
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
                case 'update':
                    console.log(`[Simulator] Step ${message.step}: ${JSON.stringify(message.message, null, 4)}`);
                    break;
                case 'log':
                default:
                    console.log(`[Simulator] Log From Server: ${JSON.stringify(message.message, null, 4)}`);
                    break;
            }
        };
    }
    startServer() {
        this.serverProcess.stdin.write('start\n');
        console.log('[Simulator] Server started');
    }
    sendWeights(ids) {
        this.serverProcess.stdin.write(JSON.stringify({ ids }) + '\n');
    }
}
new Simulator().setup();
//# sourceMappingURL=simulator.js.map