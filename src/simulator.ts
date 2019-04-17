import * as bb from 'bluebird';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import * as inquirer from 'inquirer';
import { join } from 'path';
import { stringify } from 'querystring';
import * as shelljs from 'shelljs';
import { execAsync } from './asyncShell';

interface MessageFormat {
  type: 'train' | 'log' | 'update';
  message?: string;
  clients?: number[];
  weights_file_path?: string;
  step?: number;
}

class Simulator {
  public name: string = "test";
  public localBatchSize: number = 64;
  public localEpochs: number = 1;
  public clientsPerRound: number = 15;
  public clientCount: number = 500;
  public iterations: number = 20000;
  public concurrency: number = 15;
  public serverProcess: ChildProcessWithoutNullStreams;

  public setup = async (): Promise<Simulator> => {
    // Sanity Checks

    // Prompt for Settings
    // const {
    //   name,
    //   localBatchSize,
    //   localEpochs,
    //   clientsPerRound,
    //   clientCount,
    //   iterations,
    // } = await inquirer.prompt([
    //   {
    //     type: 'input',
    //     name: 'name',
    //     message: 'What is the name of this simulation?',
    //     default: 'default',
    //   },
    //   {
    //     type: 'input',
    //     name: 'localBatchSize',
    //     message: 'What is the local batch size?',
    //     default: 64,
    //   },
    //   {
    //     type: 'input',
    //     name: 'localEpochs',
    //     message: 'What is the local epochs?',
    //     default: 2,
    //   },
    //   {
    //     type: 'input',
    //     name: 'clientsPerRound',
    //     message: 'How many clients per round?',
    //     default: 10,
    //   },
    //   {
    //     type: 'input',
    //     name: 'clientCount',
    //     message: 'How many clients in total?',
    //     default: 300,
    //   },
    //   {
    //     type: 'input',
    //     name: 'iterations',
    //     message: 'How many iterations in total?',
    //     default: 10000,
    //   },
    // ]);
    // this.name = name;
    // this.localBatchSize = localBatchSize;
    // this.localEpochs = localEpochs;
    // this.clientsPerRound = clientsPerRound;
    // this.clientCount = clientCount;
    // this.iterations = iterations;
    // Run Setup Script
    const { stderr, stdout, code } = shelljs.exec(
      `python python/setup.py -n ${this.name} -b ${this.localBatchSize} -e ${
      this.localEpochs
      } -t ${this.clientCount}`
    );
    if (code !== 0) {
      console.error('[Simulator] Setup failed');
      process.exit(1);
    }
    console.log('[Simulator] Finished setting up');

    this.serverProcess = spawn('python', [
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
  };

  private startServer() {
    this.serverProcess.stdin.write('start\n');
    console.log('[Simulator] Server started');
  }

  private sendWeights(ids: number[]) {
    this.serverProcess.stdin.write(JSON.stringify({ ids }) + '\n');
  }

  private handleServerMessage = (data: Buffer) => {
    const message: MessageFormat = JSON.parse(data.toString());

    switch (message.type) {
      case 'train':
        bb.map<number, { stdout: string; stderr: string; code: number }>(
          message.clients,
          clientId => {
            return execAsync(
              `python python/client.py -n ${this.name} -w ${
              message.weights_file_path
              } -i ${clientId} -s ${message.step}`,
              {
                async: true,
                silent: true,
              }
            );
          },
          {
            concurrency: this.concurrency,
          }
        ).then(results => {
          // Filter Results
          try {
            const successfulClients: number[] = results
              .filter(value => value.code === 0)
              .map(
                value => JSON.parse(value.stdout.split('\n').slice(-1)[0]).id
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
      case 'update':
        console.log(
          `[Simulator] Step ${message.step}: ${JSON.stringify(
            message.message,
            null,
            4
          )}`
        );
        break;
      case 'log':
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
