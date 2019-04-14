module.exports = {
    apps : [{
      name: 'FLSimulator',
      script: 'dist/simulate.js',
      args: '',
      instances: 1,
      autorestart: false,
      watch: false,
      max_memory_restart: '3G',
      env: {
        NODE_ENV: 'development'
      },
      env_production: {
        NODE_ENV: 'production'
      }
    }]
  };