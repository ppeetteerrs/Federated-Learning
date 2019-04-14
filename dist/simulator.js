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
const child_process_1 = require("child_process");
const inquirer = require("inquirer");
const shelljs = require("shelljs");
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
                    default: 32,
                },
                {
                    type: 'input',
                    name: 'localEpochs',
                    message: 'What is the local epochs?',
                    default: 1,
                },
                {
                    type: 'input',
                    name: 'clientsPerRound',
                    message: 'How many clients per round?',
                    default: 1,
                },
                {
                    type: 'input',
                    name: 'clientCount',
                    message: 'How many clients in total?',
                    default: 10,
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
            yield shelljs.exec(`python python/setup.py -n ${this.name} -b ${this.localBatchSize} -e ${this.localEpochs} -t ${this.clientCount}`);
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
                    const { stdout, stderr } = shelljs.exec(`python python/client.py -n ${this.name} -w ${message.weights_file_path} -i ${message.clients[0]}`, {
                        async: false,
                        silent: false,
                    });
                    console.log('Client Trained');
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
}
new Simulator().setup();
//# sourceMappingURL=simulator.js.map