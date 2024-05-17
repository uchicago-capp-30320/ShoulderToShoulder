import { Component, OnInit } from '@angular/core';

/**
 * Defines the loading animation component.
 *
 * This component is used to display a loading animation with dots that
 * appear and disappear in a loop.
 *
 * @example
 * ```
 * <app-loader></app-loader>
 * ```
 */
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
    }, 500);
  }
}
