class BaseRegressor {
    constructor(type) {
        this.type = type;
        this.params = { a: 0.5, b: 0.5, c: 0.5, d: 0.5 };
        this.lr = 0.1;
        this.iters = 3000;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    predict(x) {
        const { a, b, c, d } = this.params;
        switch (this.type) {
            case 'Linear': return a * x + b;
            case 'Polynomial': return a * Math.pow(x - 0.5, 2) + b * x + c;
            case 'Exponential': return a * Math.exp(b * x) + c;
            case 'Logarithmic': return a + b * Math.log(x + 0.05);
            case 'Periodic': return a * Math.sin(b * x + c) + d;
            case 'Step': return x < a ? b : c;
            case 'Logistic': return this.sigmoid((x - a) * b * 10);
        }
    }

    fit(points) {
        if (this.type === 'Step') {
            this.fitStep(points);
            return;
        }

        this.params = { a: 0.2, b: 1, c: 0.5, d: 0.5 };
        if (this.type === 'Periodic') this.params.b = 15;
        if (this.type === 'Logistic') { this.params.a = 0.5; this.params.b = 5; }

        for (let i = 0; i < this.iters; i++) {
            let da = 0, db = 0, dc = 0, dd = 0;
            for (let p of points) {
                const x = p.x / 600;
                const y = (250 - p.y) / 250;
                const pred = this.predict(x);
                const err = pred - y;

                if (this.type === 'Linear') { da += err * x; db += err; }
                else if (this.type === 'Polynomial') { da += err * Math.pow(x - 0.5, 2); db += err * x; dc += err; }
                else if (this.type === 'Exponential') {
                    da += err * Math.exp(this.params.b * x);
                    db += err * this.params.a * x * Math.exp(this.params.b * x);
                    dc += err;
                } 
                else if (this.type === 'Logarithmic') { da += err; db += err * Math.log(x + 0.05); }
                else if (this.type === 'Periodic') {
                    da += err * Math.sin(this.params.b * x + this.params.c);
                    db += err * this.params.a * x * Math.cos(this.params.b * x + this.params.c);
                    dc += err * this.params.a * Math.cos(this.params.b * x + this.params.c);
                    dd += err;
                }
                else if (this.type === 'Logistic') {
                    const s = pred * (1 - pred);
                    da += err * s * (-this.params.b * 10);
                    db += err * s * (x - this.params.a) * 10;
                }
            }
            const n = points.length;
            this.params.a -= (da / n) * this.lr;
            this.params.b -= (db / n) * this.lr;
            this.params.c -= (dc / n) * this.lr;
            this.params.d -= (dd / n) * this.lr;
        }
    }

    fitStep(points) {
        let bestA = 0.5, bestB = 0, bestC = 0;
        let minError = Infinity;

        for (let testA = 0.1; testA < 0.9; testA += 0.05) {
            const leftPoints = points.filter(p => (p.x / 600) < testA);
            const rightPoints = points.filter(p => (p.x / 600) >= testA);

            if (leftPoints.length === 0 || rightPoints.length === 0) continue;

            const avgB = leftPoints.reduce((sum, p) => sum + (250 - p.y) / 250, 0) / leftPoints.length;
            const avgC = rightPoints.reduce((sum, p) => sum + (250 - p.y) / 250, 0) / rightPoints.length;

            let totalErr = 0;
            points.forEach(p => {
                const x = p.x / 600;
                const y = (250 - p.y) / 250;
                const pred = x < testA ? avgB : avgC;
                totalErr += Math.pow(pred - y, 2);
            });

            if (totalErr < minError) {
                minError = totalErr;
                bestA = testA;
                bestB = avgB;
                bestC = avgC;
            }
        }
        this.params = { a: bestA, b: bestB, c: bestC };
    }
}

class ModelViz {
    constructor(type, formula, container) {
        this.type = type;
        const temp = document.getElementById('model-template');
        const clone = temp.content.cloneNode(true);
        this.el = clone.querySelector('.model-card');
        this.canvas = clone.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.spreadInput = this.el.querySelector('.spread-input');
        this.outlierInput = this.el.querySelector('.outlier-input');
        this.rerollBtn = this.el.querySelector('.reroll-btn');
        this.el.querySelector('.model-title').textContent = type;
        this.el.querySelector('.model-formula').textContent = formula;
        container.appendChild(this.el);
        this.model = new BaseRegressor(type);
        this.initEvents();
        this.generate();
    }

    initEvents() {
        this.rerollBtn.onclick = () => this.generate();
        this.spreadInput.oninput = () => this.generate();
        this.outlierInput.oninput = () => this.generate();
    }

    generate() {
        const spread = this.spreadInput.value / 100;
        const outlierChance = this.outlierInput.value / 100;
        this.points = [];
        
        const mode = ['clusters', 'gap', 'hetero', 'normal'][Math.floor(Math.random() * 4)];
        const baseA = Math.random() * 0.7 + 0.1;
        const baseB = Math.random() * 10 + 2;

        for (let i = 0; i < 70; i++) {
            let xNorm = Math.random();
            if (mode === 'gap' && xNorm > 0.4 && xNorm < 0.6) xNorm += 0.25; 
            if (mode === 'clusters') xNorm = Math.floor(xNorm * 5) / 5 + (Math.random() * 0.05);

            let yNorm;
            switch(this.type) {
                case 'Linear': yNorm = baseA * xNorm + 0.2; break;
                case 'Polynomial': yNorm = 2.5 * Math.pow(xNorm - 0.5, 2) + 0.2; break;
                case 'Exponential': yNorm = 0.1 * Math.exp(2 * xNorm) + 0.1; break;
                case 'Logarithmic': yNorm = 0.5 + 0.2 * Math.log(xNorm + 0.01); break;
                case 'Periodic': yNorm = 0.2 * Math.sin(baseB * xNorm) + 0.5; break;
                case 'Step': yNorm = xNorm < 0.5 ? 0.2 : 0.8; break;
                case 'Logistic': 
                    const threshold = 0.5 + (Math.random() - 0.5) * 0.2;
                    yNorm = xNorm < threshold ? 0.1 : 0.9;
                    break;
            }

            if (Math.random() < outlierChance) {
                yNorm = Math.random();
            } else {
                const noiseScale = (mode === 'hetero') ? (xNorm * spread) : spread;
                yNorm += (Math.random() - 0.5) * noiseScale;
            }

            yNorm = Math.max(0, Math.min(1, yNorm));
            this.points.push({ x: xNorm * 600, y: 250 - (yNorm * 250) });
        }
        this.model.fit(this.points);
        this.draw();
    }

    draw() {
        this.ctx.clearRect(0, 0, 600, 250);
        this.points.forEach(p => {
            this.ctx.fillStyle = '#58a6ff';
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.strokeStyle = '#ff7b72';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        for (let x = 0; x <= 600; x += 2) {
            const yNorm = this.model.predict(x / 600);
            this.ctx.lineTo(x, 250 - (yNorm * 250));
        }
        this.ctx.stroke();
    }
}

const container = document.getElementById('models-container');
const types = [
    ['Linear', 'y = ax + b'],
    ['Polynomial', 'y = a(x-0.5)² + bx + c'],
    ['Exponential', 'y = ae^(bx) + c'],
    ['Logarithmic', 'y = a + b * ln(x)'],
    ['Periodic', 'y = a * sin(bx + c) + d'],
    ['Step', 'y = (x < a) ? b : c'],
    ['Logistic', 'y = 1 / (1 + e^-z)']
];
types.forEach(t => new ModelViz(t[0], t[1], container));