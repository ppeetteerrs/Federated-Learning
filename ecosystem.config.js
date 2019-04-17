const dataset = "default";

module.exports = {
    apps : [{
      name: 'Test1',
      script: 'dist/simulator.js',
      args: [
        "--name", "Test1",
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
      name: 'Test5',
      script: 'dist/simulator.js',
      args: [
        "--name", "Test5",
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
      name: 'Test10',
      script: 'dist/simulator.js',
      args: [
        "--name", "Test10",
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
      name: 'Test15',
      script: 'dist/simulator.js',
      args: [
        "--name", "Test15",
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
    }]
  };