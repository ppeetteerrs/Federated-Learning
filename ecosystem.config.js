const dataset = "1000-clients-1-class-no-repeat";

module.exports = {
    apps : [{
      name: '1000-clients-15-per-round-1-class-1-epoch',
      script: 'dist/simulator.js',
      args: [
        "--name", "1000-clients-15-per-round-1-class-1-epoch",
        "--dataset", dataset, 
        "--gpu", 0,
        "--batchSize", 64,
        "--epochs", 1,
        "--clientsPerRound", 15,
        "--concurrency", 3,
        "--serverGPUFraction", 0.2,
        "--clientGPUFraction", 0.15,
        "--iterations", 2000
      ],
      instances: 1,
      autorestart: false,
      watch: false
    },{
      name: '1000-clients-15-per-round-1-class-5-epoch',
      script: 'dist/simulator.js',
      args: [
        "--name", "1000-clients-15-per-round-1-class-5-epoch",
        "--dataset", dataset, 
        "--gpu", 1,
        "--batchSize", 64,
        "--epochs", 5,
        "--clientsPerRound", 15,
        "--concurrency", 3,
        "--serverGPUFraction", 0.2,
        "--clientGPUFraction", 0.15,
        "--iterations", 2000
      ],
      instances: 1,
      autorestart: false,
      watch: false
    },{
      name: '1000-clients-15-per-round-1-class-10-epoch',
      script: 'dist/simulator.js',
      args: [
        "--name", "1000-clients-15-per-round-1-class-10-epoch",
        "--dataset", dataset, 
        "--gpu", 2,
        "--batchSize", 64,
        "--epochs", 10,
        "--clientsPerRound", 15,
        "--concurrency", 3,
        "--serverGPUFraction", 0.2,
        "--clientGPUFraction", 0.15,
        "--iterations", 2000
      ],
      instances: 1,
      autorestart: false,
      watch: false
    },{
      name: '1000-clients-15-per-round-1-class-15-epoch',
      script: 'dist/simulator.js',
      args: [
        "--name", "1000-clients-15-per-round-1-class-15-epoch",
        "--dataset", dataset, 
        "--gpu", 3,
        "--batchSize", 64,
        "--epochs", 15,
        "--clientsPerRound", 15,
        "--concurrency", 3,
        "--serverGPUFraction", 0.2,
        "--clientGPUFraction", 0.15,
        "--iterations", 2000
      ],
      instances: 1,
      autorestart: false,
      watch: false
    },{
      name: "tb",
      script: "./launch_tb.sh",
      interpreter: "bash"
  }]
  };