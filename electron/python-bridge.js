/**
 * Python Bridge - Optimized IPC communication with Python server
 *
 * This module provides efficient communication between Electron and the Python server,
 * with connection pooling, health checks, and automatic recovery.
 */

const http = require('http');
const log = require('electron-log');

class PythonBridge {
  constructor(serverPort) {
    this.serverPort = serverPort;
    this.baseUrl = `http://127.0.0.1:${serverPort}`;
    this.isHealthy = false;
    this.healthCheckInterval = null;
    this.requestQueue = [];
    this.maxQueueSize = 100;

    // HTTP agent for connection pooling
    this.agent = new http.Agent({
      keepAlive: true,
      keepAliveMsecs: 30000,
      maxSockets: 10,
      maxFreeSockets: 5,
      timeout: 60000
    });
  }

  /**
   * Start health monitoring
   */
  startHealthCheck() {
    // Initial health check
    this.checkHealth();

    // Periodic health checks every 30 seconds
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, 30000);
  }

  /**
   * Stop health monitoring
   */
  stopHealthCheck() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  /**
   * Check if Python server is healthy
   */
  async checkHealth() {
    try {
      const response = await this.request('/status', 'GET');
      this.isHealthy = response.statusCode === 200;

      if (this.isHealthy) {
        log.debug('Python server health check: OK');
      } else {
        log.warn('Python server health check: Failed', response.statusCode);
      }

      return this.isHealthy;
    } catch (error) {
      log.error('Python server health check error:', error.message);
      this.isHealthy = false;
      return false;
    }
  }

  /**
   * Make an optimized HTTP request to Python server
   */
  async request(endpoint, method = 'GET', data = null, options = {}) {
    return new Promise((resolve, reject) => {
      const url = new URL(endpoint, this.baseUrl);

      const requestOptions = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: method,
        agent: this.agent,
        timeout: options.timeout || 30000,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers
        }
      };

      // Add content length for POST/PUT requests
      if (data && (method === 'POST' || method === 'PUT')) {
        const body = JSON.stringify(data);
        requestOptions.headers['Content-Length'] = Buffer.byteLength(body);
      }

      const req = http.request(requestOptions, (res) => {
        let responseData = '';

        res.on('data', (chunk) => {
          responseData += chunk;
        });

        res.on('end', () => {
          try {
            const result = {
              statusCode: res.statusCode,
              headers: res.headers,
              data: responseData ? JSON.parse(responseData) : null
            };
            resolve(result);
          } catch (parseError) {
            resolve({
              statusCode: res.statusCode,
              headers: res.headers,
              data: responseData
            });
          }
        });
      });

      req.on('error', (error) => {
        log.error('Request error:', error.message);
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      // Send data for POST/PUT requests
      if (data && (method === 'POST' || method === 'PUT')) {
        req.write(JSON.stringify(data));
      }

      req.end();
    });
  }

  /**
   * Get server status
   */
  async getStatus() {
    try {
      const response = await this.request('/status', 'GET');
      return response.data;
    } catch (error) {
      log.error('Failed to get server status:', error);
      return null;
    }
  }

  /**
   * Get cluster data (optimized for frequent access)
   */
  async getClusters() {
    try {
      const response = await this.request('/', 'GET');
      return response;
    } catch (error) {
      log.error('Failed to get clusters:', error);
      return null;
    }
  }

  /**
   * Trigger feed refresh
   */
  async refreshFeeds(feedsData) {
    try {
      const response = await this.request('/refresh', 'POST', feedsData, {
        timeout: 300000 // 5 minute timeout for feed processing
      });
      return response;
    } catch (error) {
      log.error('Failed to refresh feeds:', error);
      throw error;
    }
  }

  /**
   * Summarize URL
   */
  async summarizeUrl(url, style = 'default') {
    try {
      const response = await this.request('/api/summarize', 'POST', {
        url: url,
        style: style
      }, {
        timeout: 60000 // 1 minute timeout
      });
      return response.data;
    } catch (error) {
      log.error('Failed to summarize URL:', error);
      throw error;
    }
  }

  /**
   * Get bookmarks
   */
  async getBookmarks(filters = {}) {
    try {
      const queryParams = new URLSearchParams(filters).toString();
      const endpoint = `/api/bookmarks${queryParams ? '?' + queryParams : ''}`;
      const response = await this.request(endpoint, 'GET');
      return response.data;
    } catch (error) {
      log.error('Failed to get bookmarks:', error);
      return null;
    }
  }

  /**
   * Add bookmark
   */
  async addBookmark(bookmarkData) {
    try {
      const response = await this.request('/api/bookmarks', 'POST', bookmarkData);
      return response.data;
    } catch (error) {
      log.error('Failed to add bookmark:', error);
      throw error;
    }
  }

  /**
   * Export bookmarks
   */
  async exportBookmarks(format = 'json', filters = {}) {
    try {
      const queryParams = new URLSearchParams(filters).toString();
      const endpoint = `/api/bookmarks/export/${format}${queryParams ? '?' + queryParams : ''}`;
      const response = await this.request(endpoint, 'GET');
      return response.data;
    } catch (error) {
      log.error('Failed to export bookmarks:', error);
      throw error;
    }
  }

  /**
   * Batch operations with rate limiting
   */
  async batchRequest(requests, maxConcurrent = 5) {
    const results = [];
    const executing = [];

    for (const req of requests) {
      const promise = this.request(req.endpoint, req.method, req.data, req.options)
        .then(result => {
          results.push({ success: true, data: result });
        })
        .catch(error => {
          results.push({ success: false, error: error.message });
        });

      executing.push(promise);

      if (executing.length >= maxConcurrent) {
        await Promise.race(executing);
        executing.splice(executing.findIndex(p => p === promise), 1);
      }
    }

    await Promise.all(executing);
    return results;
  }

  /**
   * Cleanup and close connections
   */
  destroy() {
    this.stopHealthCheck();
    if (this.agent) {
      this.agent.destroy();
    }
  }
}

module.exports = PythonBridge;
