import * as iq from "inquirer";
import * as shelljs from "shelljs";

async function promptForOptions() {
    return await iq.prompt<{
        name: string,
        clientCount: number,
        minSamples: number,
        maxSamples: number,
        minClasses: number,
        maxClasses: number,
        repeat: boolean
    }>([
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
}

async function generateDataset() {

    const options = await promptForOptions();

    shelljs.env.TF_CPP_MIN_LOG_LEVEL = "3";

    const { stderr, stdout, code } = shelljs.exec(
        [
            "python python/setup.py",
            "-n", options.name,
            "-t", options.clientCount,
            "-s", options.minSamples,
            "-u", options.maxSamples,
            "-v", options.minClasses,
            "-w", options.maxClasses,
            "-r", options.repeat
        ].join(" ")
    );
    if (code !== 0) {
        console.error("[Simulator] Setup failed");
        process.exit(1);
    }
}

generateDataset();
