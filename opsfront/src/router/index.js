import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Dashboard', component: () => import('../views/Dashboard.vue') },
  { path: '/agents', name: 'Agents', component: () => import('../views/Agents.vue') },
  { path: '/topology', name: 'Topology', component: () => import('../views/Topology.vue') },
  { path: '/sessions', name: 'Sessions', component: () => import('../views/Sessions.vue') },
  { path: '/resources', name: 'Resources', component: () => import('../views/Resources.vue') },
  { path: '/alerts', name: 'Alerts', component: () => import('../views/Alerts.vue') },
  { path: '/benchmark', name: 'Benchmark', component: () => import('../views/Benchmark.vue') },
  { path: '/knowledge', name: 'Knowledge', component: () => import('../views/Knowledge.vue') },
]

export default createRouter({ history: createWebHashHistory(), routes })