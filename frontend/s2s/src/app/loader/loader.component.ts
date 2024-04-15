import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-loader',
  templateUrl: './loader.component.html',
  styleUrl: './loader.component.css'
})
export class LoaderComponent implements OnInit {
  dots = Array(5).fill(0); // Creates an array with 5 elements
  showDots = 0;

  constructor() { }

  ngOnInit(): void {
    setInterval(() => {
      this.showDots = (this.showDots + 1) % (this.dots.length + 1);
    }, 1000);
  }
}
